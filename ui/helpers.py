from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    create_embedder as _create_embedder,
)
from src.models import Document

ALLOWED_EXTENSIONS = {".md", ".txt"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    return get_project_root() / "data"


def load_dotenv_config() -> None:
    load_dotenv(dotenv_path=get_project_root() / ".env", override=False)


def list_data_files() -> list[Path]:
    data_dir = get_data_dir()
    if not data_dir.is_dir():
        return []
    files = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    return files


def load_documents_from_paths(file_paths: list[Path | str]) -> list[Document]:
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        if not path.exists() or not path.is_file():
            continue

        content = path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {
            "source": str(path.relative_to(get_project_root())),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "chars": len(content),
        }

        for key in ("category", "language", "difficulty", "author", "date"):
            env_key = f"DOC_META_{path.stem.upper()}_{key.upper()}"
            value = os.getenv(env_key)
            if value:
                metadata[key] = value

        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata=metadata,
            )
        )

    return documents


def create_embedder(provider: str):
    load_dotenv_config()
    return _create_embedder(provider, allow_mock_fallback=True)


def get_embedder_backend_name(embedder) -> str:
    return getattr(embedder, "_backend_name", embedder.__class__.__name__)


def get_default_embedding_provider() -> str:
    load_dotenv_config()
    return os.getenv(EMBEDDING_PROVIDER_ENV, "local").strip().lower() or "local"


def create_mock_llm() -> Callable[[str], str]:
    def demo_llm(prompt: str) -> str:
        preview = prompt[:400].replace("\n", " ")
        return f"[Mock LLM] Trả lời dựa trên ngữ cảnh đã retrieve.\n\nXem trước prompt: {preview}..."

    return demo_llm


def create_openai_llm() -> Callable[[str], str] | None:
    load_dotenv_config()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        def chat_llm(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Trả lời ngắn gọn, chính xác dựa trên ngữ cảnh được cung cấp.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""

        return chat_llm
    except Exception:
        return None


def has_openai_api_key() -> bool:
    load_dotenv_config()
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def compare_chunking_strategies(
    text: str,
    chunk_size: int = 200,
    overlap: int = 0,
    max_sentences: int = 3,
) -> dict[str, dict[str, Any]]:
    from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker

    strategies = {
        "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
        "by_sentences": SentenceChunker(max_sentences_per_chunk=max_sentences),
        "recursive": RecursiveChunker(chunk_size=chunk_size),
    }

    comparison: dict[str, dict[str, Any]] = {}
    for name, chunker in strategies.items():
        chunks = chunker.chunk(text)
        count = len(chunks)
        avg_length = sum(len(chunk) for chunk in chunks) / count if count else 0.0
        comparison[name] = {
            "count": count,
            "avg_length": avg_length,
            "chunks": chunks,
        }
    return comparison


def benchmark_results_to_csv(results: list[dict[str, Any]]) -> str:
    if not results:
        return "query,top1_score,top1_source,relevant\n"

    lines = ["query,top1_score,top1_source,relevant"]
    for row in results:
        query = json.dumps(row.get("query", ""), ensure_ascii=False)
        score = row.get("top1_score", "")
        source = json.dumps(row.get("top1_source", ""), ensure_ascii=False)
        relevant = "yes" if row.get("relevant") else "no"
        lines.append(f"{query},{score},{source},{relevant}")
    return "\n".join(lines) + "\n"
