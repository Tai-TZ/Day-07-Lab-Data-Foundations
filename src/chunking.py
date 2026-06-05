from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []

        # Split after sentence-ending punctuation when followed by whitespace/newline.
        # Keeps punctuation in the sentence via lookbehind.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])(?:\s+|\n+)", raw) if s.strip()]
        if not sentences:
            return [raw]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []

        seps = list(self.separators) if self.separators else [""]
        chunks = self._split(raw, seps)
        return [c.strip() for c in chunks if c and c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separators left: hard fallback to fixed-size chunks.
        if not remaining_separators:
            return FixedSizeChunker(chunk_size=self.chunk_size, overlap=0).chunk(current_text)

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        # Empty separator means "can't split" -> fixed-size.
        if sep == "":
            return FixedSizeChunker(chunk_size=self.chunk_size, overlap=0).chunk(current_text)

        pieces = current_text.split(sep)
        if len(pieces) == 1:
            # Separator not found, try next.
            return self._split(current_text, rest)

        merged: list[str] = []
        buf = ""
        for piece in pieces:
            candidate = piece if not buf else f"{buf}{sep}{piece}"

            if len(candidate) <= self.chunk_size:
                buf = candidate
                continue

            # Flush existing buffer if it has something.
            if buf:
                merged.append(buf)
                buf = ""

            # Oversized single piece: recurse with remaining separators.
            if len(piece) > self.chunk_size:
                merged.extend(self._split(piece, rest))
            else:
                buf = piece

        if buf:
            merged.append(buf)

        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    denom = math.sqrt(_dot(vec_a, vec_a)) * math.sqrt(_dot(vec_b, vec_b))
    if denom == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / denom


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        raw = text or ""

        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10)).chunk(raw)
        sentence_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(raw)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(raw)

        def _stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": _stats(fixed_chunks),
            "by_sentences": _stats(sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }
