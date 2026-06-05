from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        return self.store.search(question, top_k=top_k)

    def build_prompt(self, question: str, results: list[dict] | None = None, top_k: int = 3) -> str:
        rows = results if results is not None else self.retrieve(question, top_k=top_k)
        context_blocks: list[str] = []
        for i, row in enumerate(rows, start=1):
            content = (row.get("content") or "").strip()
            if not content:
                continue
            metadata = row.get("metadata") or {}
            source = metadata.get("source") or metadata.get("doc_id") or "unknown"
            page = metadata.get("page") or "?"
            context_blocks.append(f"[{i}] ({source}, p.{page})\n{content}")

        context = "\n\n".join(context_blocks) if context_blocks else "(no relevant context retrieved)"
        return (
            "Answer the question using ONLY the context below. Cite sources as [1], [2], etc.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    def answer(self, question: str, top_k: int = 3) -> str:
        return self.llm_fn(self.build_prompt(question, top_k=top_k))

    def answer_with_prompt(self, question: str, top_k: int = 3) -> tuple[str, str, list[dict]]:
        results = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, results=results, top_k=top_k)
        return self.llm_fn(prompt), prompt, results
