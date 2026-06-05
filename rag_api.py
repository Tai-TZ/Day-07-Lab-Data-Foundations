from __future__ import annotations

import os
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import LOCAL_EMBEDDING_MODEL, create_embedder, get_vector_dim
from src.llm import (
    LLMError,
    create_openrouter_llm,
    get_default_openrouter_model,
    get_openrouter_models,
    has_openrouter_api_key,
    resolve_openrouter_model,
)
from src.models import Document
from src.store import EmbeddingStore
from ui.helpers import load_dotenv_config

load_dotenv_config()

DATA_DIR = Path(__file__).parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {".md", ".pdf"}
ALLOWED_DATA_EXTENSIONS = {".md", ".txt", ".pdf"}

ChunkStrategy = Literal["character", "token", "sentence", "recursive"]


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
    if ext == ".pdf":
        return "PDF"
    return "TXT"


def _extract_pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Run: pip install pypdf") from exc

    reader = PdfReader(BytesIO(raw))
    pages: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)
    content = "\n\n".join(pages).strip()
    if not content:
        raise ValueError("No extractable text found in PDF (scanned/image-only PDFs are not supported).")
    return content


def _read_file_content(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path.read_bytes())
    return path.read_text(encoding="utf-8")


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    safe = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return safe or "document"


def _unique_upload_path(filename: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        ext = ".md"
    candidate = UPLOAD_DIR / f"{stem}{ext}"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = UPLOAD_DIR / f"{stem}_{counter}{ext}"
        if not candidate.exists():
            return candidate
        counter += 1


def _load_documents_from_data_dir() -> list[Document]:
    docs: list[Document] = []
    search_roots = [DATA_DIR, UPLOAD_DIR]
    seen_ids: set[str] = set()

    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in ALLOWED_DATA_EXTENSIONS or not path.is_file():
                continue
            doc_id = path.stem
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            try:
                content = _read_file_content(path)
            except Exception:
                continue
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


def _split_sentences(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [s.strip() for s in re.split(r"(?<=[.!?])(?:\s+|\n+)", raw) if s.strip()]
    return parts if parts else [raw]


def _project_2d(vec: list[float]) -> tuple[float, float]:
    if len(vec) >= 2:
        scale = max(abs(vec[0]), abs(vec[1]), 1e-6)
        return float(vec[0] / scale), float(vec[1] / scale)
    return 0.0, 0.0


def _make_chunker(strategy: ChunkStrategy, size: int, overlap: int):
    unit = 4 if strategy == "token" else 1
    char_size = max(40, size * unit)
    char_overlap = max(0, overlap * unit)

    if strategy == "sentence":
        max_sentences = max(1, min(8, size // 64 or 1))
        return SentenceChunker(max_sentences_per_chunk=max_sentences)
    if strategy == "character":
        return FixedSizeChunker(chunk_size=char_size, overlap=char_overlap)
    return RecursiveChunker(chunk_size=char_size)


def _map_sentences_to_chunks(answer: str, chunks: list["RagChunk"]) -> list[dict[str, str]]:
    mapped: list[dict[str, str]] = []
    for sentence in _split_sentences(answer):
        best_id = chunks[0].id if chunks else "chunk-0"
        best_overlap = -1
        sentence_words = [w.lower() for w in re.findall(r"\w+", sentence) if len(w) > 3]
        for chunk in chunks:
            chunk_lower = chunk.text.lower()
            overlap = sum(1 for word in sentence_words if word in chunk_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = chunk.id
        mapped.append({"text": sentence, "chunkId": best_id})
    return mapped


def _require_openrouter() -> None:
    if not has_openrouter_api_key():
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not set. Add your key to .env and restart the backend.",
        )


class RagChunk(BaseModel):
    id: str
    doc_id: str = ""
    chunk_index: int = 0
    source: str
    page: int = 1
    text: str
    score: float
    used: bool | None = None
    px: float | None = None
    py: float | None = None
    embedding_preview: list[float] | None = None


class RagDocument(BaseModel):
    id: str
    name: str
    type: str = Field(pattern="^(PDF|TXT|URL|MD|DOCX)$")
    size: str
    chunks: int
    preview: list[str]
    raw: str = ""
    uploaded: bool = False


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    llm_model: str | None = None


class QueryResponse(BaseModel):
    chunks: list[RagChunk]
    answer: str
    sentences: list[dict[str, str]]
    query_embedding: list[float] = Field(default_factory=list)
    prompt: str = ""
    llm_model: str = ""


class UploadResponse(BaseModel):
    document: RagDocument
    message: str


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    dimensions: int


class ChunkConfigRequest(BaseModel):
    strategy: ChunkStrategy = "recursive"
    size: int = 512
    overlap: int = 64


class RagCatalog:
    def __init__(self) -> None:
        self.embedder = create_embedder(allow_mock_fallback=False)
        self.vector_dim = get_vector_dim(self.embedder)
        self.embedding_model = getattr(self.embedder, "_backend_name", LOCAL_EMBEDDING_MODEL)
        self.chunk_strategy: ChunkStrategy = "recursive"
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        self.chunker = _make_chunker(self.chunk_strategy, self.chunk_size, self.chunk_overlap)
        self.store = EmbeddingStore(collection_name="rag_reveal_store", embedding_fn=self.embedder)
        self.default_llm_model = get_default_openrouter_model()
        self.llm_models = get_openrouter_models()
        self.rag_docs: list[RagDocument] = []
        self._indexed_ids: set[str] = set()
        self._source_docs: list[Document] = []

    def configure_chunking(self, strategy: ChunkStrategy, size: int, overlap: int) -> None:
        self.chunk_strategy = strategy
        self.chunk_size = max(40, size)
        self.chunk_overlap = max(0, overlap)
        self.chunker = _make_chunker(strategy, self.chunk_size, self.chunk_overlap)

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
            raw=doc.content or "",
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
        self._source_docs = _load_documents_from_data_dir()
        for doc in self._source_docs:
            self.index_document(doc)

    def reindex_all(self, strategy: ChunkStrategy, size: int, overlap: int) -> None:
        self.configure_chunking(strategy, size, overlap)
        self.store.clear()
        self.rag_docs = []
        self._indexed_ids = set()
        for doc in self._source_docs:
            self.index_document(doc)

    def list_indexed_chunks(self, doc_id: str | None = None) -> list[RagChunk]:
        rows: list[RagChunk] = []
        for record in self.store.list_records():
            md = record.get("metadata") or {}
            record_doc_id = str(md.get("doc_id") or "")
            if doc_id and record_doc_id != doc_id:
                continue
            emb = record.get("embedding") or []
            emb_list = emb if isinstance(emb, list) else []
            px, py = _project_2d(emb_list)
            rows.append(
                RagChunk(
                    id=str(record.get("id") or ""),
                    doc_id=record_doc_id,
                    chunk_index=int(md.get("chunk_index") or 0),
                    source=str(md.get("source") or record_doc_id or "unknown"),
                    page=int(md.get("page") or 1),
                    text=str(record.get("content") or ""),
                    score=0.0,
                    px=px,
                    py=py,
                    embedding_preview=[float(v) for v in emb_list[:64]],
                )
            )
        rows.sort(key=lambda c: (c.doc_id, c.chunk_index))
        return rows

    def embed_text(self, text: str) -> list[float]:
        return self.embedder(text or "")

    def query(self, query_text: str, top_k: int, llm_model: str | None = None) -> QueryResponse:
        _require_openrouter()
        selected_model = resolve_openrouter_model(llm_model or self.default_llm_model)

        models_to_try = [selected_model] + [m for m in self.llm_models if m != selected_model]
        answer = ""
        prompt = ""
        results: list[dict] = []
        last_error: LLMError | None = None

        for candidate in models_to_try:
            try:
                agent = KnowledgeBaseAgent(store=self.store, llm_fn=create_openrouter_llm(candidate))
                answer, prompt, results = agent.answer_with_prompt(query_text, top_k=top_k)
                selected_model = candidate
                last_error = None
                break
            except LLMError as exc:
                last_error = exc

        if last_error is not None:
            raise HTTPException(status_code=502, detail=str(last_error)) from last_error

        chunks: list[RagChunk] = []
        for i, r in enumerate(results):
            md = r.get("metadata") or {}
            chunks.append(
                RagChunk(
                    id=r.get("id") or f"chunk-{i}",
                    doc_id=str(md.get("doc_id") or ""),
                    chunk_index=int(md.get("chunk_index") or 0),
                    source=str(md.get("source") or md.get("doc_id") or "unknown"),
                    page=int(md.get("page") or 1),
                    text=str(r.get("content") or ""),
                    score=float(r.get("score") or 0.0),
                )
            )

        sentences = _map_sentences_to_chunks(answer, chunks)
        query_embedding = self.embed_text(query_text)
        return QueryResponse(
            chunks=chunks,
            answer=answer,
            sentences=sentences,
            query_embedding=query_embedding,
            prompt=prompt,
            llm_model=selected_model,
        )


_catalog: RagCatalog | None = None
_catalog_error: str | None = None
_catalog_lock = threading.Lock()


def _init_catalog() -> RagCatalog:
    global _catalog, _catalog_error
    if _catalog is not None:
        return _catalog
    with _catalog_lock:
        if _catalog is not None:
            return _catalog
        try:
            catalog = RagCatalog()
            catalog.load_all()
            _catalog = catalog
            _catalog_error = None
            return catalog
        except Exception as exc:
            _catalog_error = str(exc)
            raise


def get_catalog() -> RagCatalog:
    try:
        return _init_catalog()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Embedding backend unavailable: {exc}") from exc


app = FastAPI(title="Day-07 RAG API", version="1.0")
_dev_origin_regex = os.getenv(
    "RAG_CORS_ORIGIN_REGEX",
    r"http://(localhost|127\.0\.0\.1):\d+",
)
_extra_origins = [
    os.getenv("RAG_REVEAL_ORIGIN", "http://localhost:5173"),
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in _extra_origins if o],
    allow_origin_regex=_dev_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_catalog() -> None:
    def _warmup() -> None:
        try:
            _init_catalog()
        except Exception:
            pass

    threading.Thread(target=_warmup, daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    if _catalog is None:
        status = "error" if _catalog_error else "loading"
        return {
            "ok": False,
            "status": status,
            "error": _catalog_error,
            "documents": 0,
            "chunks": 0,
            "embedding_model": LOCAL_EMBEDDING_MODEL,
            "vector_dim": 384,
            "llm": "openrouter" if has_openrouter_api_key() else "unconfigured",
        }
    catalog = _catalog
    return {
        "ok": True,
        "status": "ready",
        "documents": len(catalog.rag_docs),
        "chunks": catalog.store.get_collection_size(),
        "embedding_model": catalog.embedding_model,
        "vector_dim": catalog.vector_dim,
        "llm": "openrouter" if has_openrouter_api_key() else "unconfigured",
    }


@app.get("/config")
def config() -> dict[str, Any]:
    catalog = get_catalog()
    return {
        "embedding_model": catalog.embedding_model,
        "vector_dim": catalog.vector_dim,
        "chunk_strategy": catalog.chunk_strategy,
        "chunk_size": catalog.chunk_size,
        "chunk_overlap": catalog.chunk_overlap,
        "llm": "openrouter" if has_openrouter_api_key() else "unconfigured",
        "llm_model": catalog.default_llm_model,
        "llm_models": catalog.llm_models,
        "documents": len(catalog.rag_docs),
        "chunks": catalog.store.get_collection_size(),
    }


@app.get("/documents")
def documents() -> list[dict[str, Any]]:
    catalog = get_catalog()
    return [d.model_dump() for d in catalog.rag_docs]


@app.get("/chunks")
def chunks(doc_id: str | None = None) -> list[dict[str, Any]]:
    catalog = get_catalog()
    return [c.model_dump() for c in catalog.list_indexed_chunks(doc_id=doc_id)]


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if not (req.text or "").strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    catalog = get_catalog()
    vector = catalog.embed_text(req.text)
    return EmbedResponse(
        embedding=vector,
        model=catalog.embedding_model,
        dimensions=len(vector),
    )


@app.post("/reindex")
def reindex(req: ChunkConfigRequest) -> dict[str, Any]:
    catalog = get_catalog()
    catalog.reindex_all(req.strategy, req.size, req.overlap)
    return {
        "ok": True,
        "documents": len(catalog.rag_docs),
        "chunks": catalog.store.get_collection_size(),
        "chunk_strategy": catalog.chunk_strategy,
        "chunk_size": catalog.chunk_size,
        "chunk_overlap": catalog.chunk_overlap,
    }


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = (file.filename or "").strip()
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .md and .pdf files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

    dest = _unique_upload_path(filename)
    if ext == ".pdf":
        try:
            content = _extract_pdf_text(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        dest.write_bytes(raw)
    else:
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Markdown file must be valid UTF-8 text.") from exc
        dest.write_text(content, encoding="utf-8")

    doc = Document(
        id=dest.stem,
        content=content,
        metadata={
            "source": str(dest.as_posix()),
            "extension": ext,
            "uploaded": True,
            "original_filename": filename,
        },
    )
    catalog = get_catalog()
    catalog._source_docs = [d for d in catalog._source_docs if d.id != doc.id]
    catalog._source_docs.append(doc)
    rag_doc = catalog.index_document(doc)
    return UploadResponse(
        document=rag_doc,
        message=f"Uploaded and indexed {rag_doc.name} ({rag_doc.chunks} chunks).",
    )


@app.post("/query")
def query(req: QueryRequest) -> QueryResponse:
    if not (req.query or "").strip():
        raise HTTPException(status_code=400, detail="Query is required.")
    catalog = get_catalog()
    return catalog.query(req.query, top_k=req.top_k, llm_model=req.llm_model)
