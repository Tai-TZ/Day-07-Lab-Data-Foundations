from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chunking import FixedSizeChunker
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = Path(__file__).parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    value = float(num_bytes)
    for unit in ["KB", "MB", "GB"]:
        value /= 1024
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
    return f"{num_bytes} B"


def _guess_type(ext: str) -> str:
    ext = (ext or "").lower()
    if ext == ".md":
        return "MD"
    if ext == ".txt":
        return "TXT"
    return "TXT"


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    safe = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return safe or "document"


def _unique_upload_path(filename: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(filename)
    candidate = UPLOAD_DIR / f"{stem}.md"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = UPLOAD_DIR / f"{stem}_{counter}.md"
        if not candidate.exists():
            return candidate
        counter += 1


def _load_documents_from_data_dir() -> list[Document]:
    allowed = {".md", ".txt"}
    docs: list[Document] = []
    search_roots = [DATA_DIR, UPLOAD_DIR]
    seen_ids: set[str] = set()

    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in allowed or not path.is_file():
                continue
            doc_id = path.stem
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            content = path.read_text(encoding="utf-8")
            docs.append(
                Document(
                    id=doc_id,
                    content=content,
                    metadata={
                        "source": str(path.as_posix()),
                        "extension": path.suffix.lower(),
                        "uploaded": root == UPLOAD_DIR,
                    },
                )
            )
    return docs


class RagChunk(BaseModel):
    id: str
    source: str
    page: int = 1
    text: str
    score: float
    used: bool | None = None


class RagDocument(BaseModel):
    id: str
    name: str
    type: str = Field(pattern="^(PDF|TXT|URL|MD)$")
    size: str
    chunks: int
    preview: list[str]
    uploaded: bool = False


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class QueryResponse(BaseModel):
    chunks: list[RagChunk]
    answer: str
    sentences: list[dict[str, str]]


class UploadResponse(BaseModel):
    document: RagDocument
    message: str


def _split_sentences(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [s.strip() for s in re.split(r"(?<=[.!?])(?:\s+|\n+)", raw) if s.strip()]
    return parts if parts else [raw]


class RagCatalog:
    def __init__(self) -> None:
        self.chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        self.store = EmbeddingStore(collection_name="rag_reveal_store", embedding_fn=_mock_embed)
        self.rag_docs: list[RagDocument] = []
        self._indexed_ids: set[str] = set()

    def _make_rag_document(self, doc: Document, chunks: list[str]) -> RagDocument:
        src = (doc.metadata or {}).get("source", doc.id)
        ext = (Path(src).suffix or (doc.metadata or {}).get("extension", "")).lower()
        preview = [c.strip().replace("\n", " ") for c in chunks[:5]]
        return RagDocument(
            id=doc.id,
            name=Path(src).name if src else doc.id,
            type=_guess_type(ext),
            size=_human_size(len((doc.content or "").encode("utf-8"))),
            chunks=len(chunks),
            preview=preview,
            uploaded=bool((doc.metadata or {}).get("uploaded")),
        )

    def index_document(self, doc: Document) -> RagDocument:
        if doc.id in self._indexed_ids:
            self.store.delete_document(doc.id)
            self.rag_docs = [d for d in self.rag_docs if d.id != doc.id]

        chunks = self.chunker.chunk(doc.content)
        src = (doc.metadata or {}).get("source", doc.id)
        chunk_docs: list[Document] = []

        for idx, chunk_text in enumerate(chunks):
            chunk_docs.append(
                Document(
                    id=f"{doc.id}__chunk_{idx}",
                    content=chunk_text,
                    metadata={
                        **(doc.metadata or {}),
                        "doc_id": doc.id,
                        "source": Path(src).name if src else doc.id,
                        "page": idx + 1,
                        "chunk_index": idx,
                    },
                )
            )

        if chunk_docs:
            self.store.add_documents(chunk_docs)

        rag_doc = self._make_rag_document(doc, chunks)
        self.rag_docs.append(rag_doc)
        self._indexed_ids.add(doc.id)
        return rag_doc

    def load_all(self) -> None:
        for doc in _load_documents_from_data_dir():
            self.index_document(doc)


CATALOG = RagCatalog()
CATALOG.load_all()


app = FastAPI(title="Day-07 RAG API", version="0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("RAG_REVEAL_ORIGIN", "http://localhost:5173"),
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "documents": len(CATALOG.rag_docs),
        "chunks": CATALOG.store.get_collection_size(),
    }


@app.get("/documents")
def documents() -> list[dict[str, Any]]:
    return [d.model_dump() for d in CATALOG.rag_docs]


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 5 MB limit.")

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.") from exc

    dest = _unique_upload_path(filename)
    dest.write_text(content, encoding="utf-8")

    doc = Document(
        id=dest.stem,
        content=content,
        metadata={
            "source": str(dest.as_posix()),
            "extension": ".md",
            "uploaded": True,
            "original_filename": filename,
        },
    )
    rag_doc = CATALOG.index_document(doc)
    return UploadResponse(
        document=rag_doc,
        message=f"Uploaded and indexed {rag_doc.name} ({rag_doc.chunks} chunks).",
    )


@app.post("/query")
def query(req: QueryRequest) -> QueryResponse:
    results = CATALOG.store.search(req.query, top_k=req.top_k)
    chunks: list[RagChunk] = []
    for i, r in enumerate(results):
        md = r.get("metadata") or {}
        chunks.append(
            RagChunk(
                id=r.get("id") or f"chunk-{i}",
                source=str(md.get("source") or md.get("doc_id") or "unknown"),
                page=int(md.get("page") or 1),
                text=str(r.get("content") or ""),
                score=float(r.get("score") or 0.0),
            )
        )

    answer_parts = []
    for c in chunks[: min(3, len(chunks))]:
        if c.text:
            answer_parts.append(c.text.strip().replace("\n", " "))
    answer = " ".join(answer_parts) if answer_parts else "No relevant context retrieved."

    sentences = []
    for s in _split_sentences(answer):
        sentences.append({"text": s, "chunkId": chunks[0].id if chunks else "chunk-0"})

    return QueryResponse(chunks=chunks, answer=answer, sentences=sentences)
