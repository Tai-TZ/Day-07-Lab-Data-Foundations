from __future__ import annotations

import hashlib
import math
import os
from typing import Callable, Protocol

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class Embedder(Protocol):
    _backend_name: str

    def __call__(self, text: str) -> list[float]: ...


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Local MiniLM embedder via sentence-transformers, with fastembed ONNX fallback."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._backend = "sentence-transformers"
        self._backend_name = model_name
        self._vector_dim = 384
        self._use_fastembed = False
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self._vector_dim = int(self.model.get_sentence_embedding_dimension())
            return
        except Exception:
            pass

        try:
            from fastembed import TextEmbedding

            fastembed_name = (
                model_name
                if model_name.startswith("sentence-transformers/")
                else f"sentence-transformers/{model_name}"
            )
            self.model = TextEmbedding(model_name=fastembed_name)
            self._use_fastembed = True
            self._backend = "fastembed"
            self._backend_name = f"{model_name} (onnx)"
            probe = list(self.model.embed(["dimension probe"]))[0]
            self._vector_dim = len(probe)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load local embedder. Install: pip install sentence-transformers fastembed"
            ) from exc

    def __call__(self, text: str) -> list[float]:
        if self._use_fastembed:
            vector = list(self.model.embed([text or ""]))[0]
            values = [float(value) for value in vector]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            return [value / norm for value in values]

        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]

    @property
    def vector_dim(self) -> int:
        return self._vector_dim


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


_mock_embed = MockEmbedder()


def create_embedder(provider: str | None = None, *, allow_mock_fallback: bool = False) -> Embedder:
    """Create embedding backend from provider name or EMBEDDING_PROVIDER env."""
    selected = (provider or os.getenv(EMBEDDING_PROVIDER_ENV, "local")).strip().lower()

    if selected == "local":
        try:
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
            return LocalEmbedder(model_name=model_name)
        except Exception as exc:
            if allow_mock_fallback:
                return _mock_embed
            raise RuntimeError(
                "Failed to load local embedder. Install: pip install sentence-transformers fastembed"
            ) from exc

    if selected == "openai":
        try:
            model_name = os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
            return OpenAIEmbedder(model_name=model_name)
        except Exception as exc:
            if allow_mock_fallback:
                return _mock_embed
            raise RuntimeError("Failed to load OpenAI embedder. Set OPENAI_API_KEY.") from exc

    if selected == "mock":
        return _mock_embed

    if allow_mock_fallback:
        return _mock_embed
    raise ValueError(f"Unknown embedding provider: {selected}")


def get_vector_dim(embedder: Embedder) -> int:
    dim = getattr(embedder, "vector_dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    probe = embedder("dimension probe")
    return len(probe)
