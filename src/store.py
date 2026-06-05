from __future__ import annotations

import os
from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        # ChromaDB is opt-in via CHROMA_PERSIST_DIR so tests and local runs
        # use a clean in-memory store unless persistence is explicitly requested.
        persist_dir = os.getenv("CHROMA_PERSIST_DIR")
        if persist_dir:
            try:
                import chromadb

                client = chromadb.PersistentClient(path=persist_dir)
                self._collection = client.get_or_create_collection(name=self._collection_name)
                self._use_chroma = True
            except Exception:
                self._use_chroma = False
                self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content or "")
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)

        record_id = f"{doc.id}:{self._next_index}"
        self._next_index += 1

        return {
            "id": record_id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        q_emb = self._embedding_fn(query or "")
        scored: list[dict[str, Any]] = []
        for r in records:
            score = float(compute_similarity(q_emb, r["embedding"]))
            scored.append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(0, int(top_k))]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma and self._collection is not None:
            ids: list[str] = []
            documents: list[str] = []
            embeddings: list[list[float]] = []
            metadatas: list[dict[str, Any]] = []
            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            return

        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            q_emb = self._embedding_fn(query or "")
            res = self._collection.query(
                query_embeddings=[q_emb],
                n_results=max(0, int(top_k)),
                include=["documents", "metadatas", "distances"],
            )

            # Chroma returns lists-of-lists per query (we have one query).
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            distances = (res.get("distances") or [[]])[0]

            out: list[dict[str, Any]] = []
            for content, metadata, distance in zip(docs, metas, distances):
                # Convert distance to a "higher-is-better" score (rough heuristic).
                score = 1.0 / (1.0 + float(distance))
                out.append({"content": content, "metadata": metadata or {}, "score": score})
            out.sort(key=lambda x: x["score"], reverse=True)
            return out[: max(0, int(top_k))]

        return self._search_records(query=query, records=self._store, top_k=top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:
                return 0
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            q_emb = self._embedding_fn(query or "")
            res = self._collection.query(
                query_embeddings=[q_emb],
                n_results=max(0, int(top_k)),
                where=metadata_filter,
                include=["documents", "metadatas", "distances"],
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            distances = (res.get("distances") or [[]])[0]
            out: list[dict[str, Any]] = []
            for content, metadata, distance in zip(docs, metas, distances):
                score = 1.0 / (1.0 + float(distance))
                out.append({"content": content, "metadata": metadata or {}, "score": score})
            out.sort(key=lambda x: x["score"], reverse=True)
            return out[: max(0, int(top_k))]

        filtered = []
        for r in self._store:
            md = r.get("metadata") or {}
            if all(md.get(k) == v for k, v in metadata_filter.items()):
                filtered.append(r)
        return self._search_records(query=query, records=filtered, top_k=top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            try:
                # Chroma delete is best-effort; report True if collection size changes.
                before = self.get_collection_size()
                self._collection.delete(where={"doc_id": doc_id})
                after = self.get_collection_size()
                return after < before
            except Exception:
                return False

        before = len(self._store)
        self._store = [r for r in self._store if (r.get("metadata") or {}).get("doc_id") != doc_id]
        return len(self._store) < before

    def clear(self) -> None:
        """Remove all stored chunks from the in-memory backend."""
        self._store = []
        self._next_index = 0
        if self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(where={})
            except Exception:
                pass

    def list_records(self) -> list[dict[str, Any]]:
        """Return all indexed chunk records (including embeddings)."""
        if self._use_chroma and self._collection is not None:
            try:
                result = self._collection.get(include=["documents", "metadatas", "embeddings"])
                records: list[dict[str, Any]] = []
                ids = result.get("ids") or []
                docs = result.get("documents") or []
                metas = result.get("metadatas") or []
                embs = result.get("embeddings") or []
                for i, record_id in enumerate(ids):
                    records.append(
                        {
                            "id": record_id,
                            "content": docs[i] if i < len(docs) else "",
                            "metadata": metas[i] if i < len(metas) else {},
                            "embedding": embs[i] if i < len(embs) else [],
                        }
                    )
                return records
            except Exception:
                return list(self._store)
        return list(self._store)
