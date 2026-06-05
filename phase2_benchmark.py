"""
Phase 2 benchmark runner — terminal-friendly, no UI.

Usage:
    py phase2_benchmark.py
    py phase2_benchmark.py --export report/phase2_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from typing import Any, Callable

from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
    compute_similarity,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

DOCUMENT_CATALOG: list[dict[str, Any]] = [
    {
        "path": "customer_support_playbook.txt",
        "category": "support",
        "language": "en",
        "doc_type": "playbook",
        "audience": "internal",
    },
    {
        "path": "python_intro.txt",
        "category": "technical",
        "language": "en",
        "doc_type": "tutorial",
        "audience": "public",
    },
    {
        "path": "vector_store_notes.md",
        "category": "technical",
        "language": "en",
        "doc_type": "notes",
        "audience": "public",
    },
    {
        "path": "rag_system_design.md",
        "category": "technical",
        "language": "en",
        "doc_type": "design",
        "audience": "internal",
    },
    {
        "path": "chunking_experiment_report.md",
        "category": "technical",
        "language": "en",
        "doc_type": "report",
        "audience": "internal",
    },
    {
        "path": "vi_retrieval_notes.md",
        "category": "technical",
        "language": "vi",
        "doc_type": "notes",
        "audience": "internal",
    },
]

BASELINE_DOCS = [
    "customer_support_playbook.txt",
    "chunking_experiment_report.md",
    "vi_retrieval_notes.md",
]

BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {
        "id": 1,
        "query": "What should support articles avoid writing?",
        "gold_answer": (
            "Support articles should avoid vague statements such as "
            "'check the settings' or 'contact engineering if needed.' "
            "They should specify the exact page, button, or log source."
        ),
        "expected_source": "customer_support_playbook",
        "filter": None,
    },
    {
        "id": 2,
        "query": "What is the proposed architecture for the RAG system?",
        "gold_answer": (
            "Ingestion chunks documents and stores segments with metadata; "
            "retrieval embeds questions and applies optional metadata filters; "
            "the application layer builds a prompt from top retrieved chunks."
        ),
        "expected_source": "rag_system_design",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Which chunking strategy performed best in the experiment?",
        "gold_answer": (
            "Recursive chunking offered the best balance: it preserved context "
            "while staying within the target size range."
        ),
        "expected_source": "chunking_experiment_report",
        "filter": None,
    },
    {
        "id": 4,
        "query": "Metadata giúp retrieval tránh nhầm tài liệu như thế nào?",
        "gold_answer": (
            "Metadata (phòng ban, ngôn ngữ, độ nhạy cảm, ngày cập nhật) giúp "
            "lọc tài liệu phù hợp, ví dụ tránh lấy marketing hoặc tài liệu tiếng Anh "
            "khi hỏi về tài liệu kỹ thuật tiếng Việt."
        ),
        "expected_source": "vi_retrieval_notes",
        "filter": {"language": "vi"},
    },
    {
        "id": 5,
        "query": "What are the four stages of a vector search pipeline?",
        "gold_answer": (
            "Chunk documents, embed each chunk, store the vector and metadata, "
            "then embed the query and rank stored vectors by similarity."
        ),
        "expected_source": "vector_store_notes",
        "filter": None,
    },
]

SIMILARITY_PAIRS: list[dict[str, str]] = [
    {
        "a": "Python is widely used for machine learning and data analysis.",
        "b": "Teams use Python for ML workflows and data science tasks.",
        "prediction": "high",
    },
    {
        "a": "Vector stores retrieve similar embeddings for semantic search.",
        "b": "A vector database ranks chunks by similarity to the query.",
        "prediction": "high",
    },
    {
        "a": "Customer support articles should use specific troubleshooting steps.",
        "b": "The billing API deployment requires Kubernetes credentials.",
        "prediction": "low",
    },
    {
        "a": "Chunking splits documents into smaller retrieval units.",
        "b": "Recursive chunking tries paragraph boundaries before smaller splits.",
        "prediction": "high",
    },
    {
        "a": "Python is a programming language used for automation.",
        "b": "Chocolate cake recipes need flour, eggs, and sugar.",
        "prediction": "low",
    },
]

STRATEGIES: dict[str, Callable[[], Any]] = {
    "fixed_size_300": lambda: FixedSizeChunker(chunk_size=300, overlap=50),
    "sentence_2": lambda: SentenceChunker(max_sentences_per_chunk=2),
    "recursive_300": lambda: RecursiveChunker(chunk_size=300),
}


def load_source_documents() -> list[Document]:
    documents: list[Document] = []
    for entry in DOCUMENT_CATALOG:
        path = DATA_DIR / entry["path"]
        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={
                    "source": str(path.relative_to(ROOT)),
                    "category": entry["category"],
                    "language": entry["language"],
                    "doc_type": entry["doc_type"],
                    "audience": entry["audience"],
                },
            )
        )
    return documents


def chunk_documents(source_docs: list[Document], chunker: Any) -> list[Document]:
    chunked: list[Document] = []
    for doc in source_docs:
        for index, chunk_text in enumerate(chunker.chunk(doc.content)):
            chunked.append(
                Document(
                    id=f"{doc.id}__chunk_{index}",
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "parent_doc_id": doc.id,
                        "chunk_index": index,
                    },
                )
            )
    return chunked


def summarize(text: str, limit: int = 120) -> str:
    return " ".join(text.split())[:limit]


def source_from_metadata(metadata: dict[str, Any]) -> str:
    parent = metadata.get("parent_doc_id") or metadata.get("doc_id") or ""
    if parent:
        return str(parent)
    source = metadata.get("source", "")
    return Path(source).stem if source else "unknown"


def run_baseline_analysis() -> dict[str, Any]:
    comparator = ChunkingStrategyComparator()
    results: dict[str, Any] = {}
    for filename in BASELINE_DOCS:
        text = (DATA_DIR / filename).read_text(encoding="utf-8")
        comparison = comparator.compare(text, chunk_size=200)
        results[filename] = {
            strategy: {
                "count": stats["count"],
                "avg_length": round(stats["avg_length"], 1),
            }
            for strategy, stats in comparison.items()
        }
    return results


def evaluate_query(
    store: EmbeddingStore,
    agent: KnowledgeBaseAgent,
    item: dict[str, Any],
    top_k: int = 3,
) -> dict[str, Any]:
    if item["filter"]:
        results = store.search_with_filter(item["query"], top_k=top_k, metadata_filter=item["filter"])
    else:
        results = store.search(item["query"], top_k=top_k)

    top3_sources = [source_from_metadata(result["metadata"]) for result in results]
    expected = item["expected_source"]
    relevant_in_top3 = expected in top3_sources

    top1 = results[0] if results else None
    answer = agent.answer(item["query"], top_k=top_k)

    return {
        "id": item["id"],
        "query": item["query"],
        "gold_answer": item["gold_answer"],
        "expected_source": expected,
        "relevant_in_top3": relevant_in_top3,
        "top3_sources": top3_sources,
        "top1_preview": summarize(top1["content"]) if top1 else "",
        "top1_score": round(top1["score"], 3) if top1 else None,
        "agent_answer_preview": summarize(answer, 180),
    }


def run_strategy_benchmark(strategy_name: str, source_docs: list[Document]) -> dict[str, Any]:
    chunker = STRATEGIES[strategy_name]()
    store = EmbeddingStore(collection_name=f"phase2_{strategy_name}", embedding_fn=_mock_embed)
    store.add_documents(chunk_documents(source_docs, chunker))

    agent = KnowledgeBaseAgent(store=store, llm_fn=lambda prompt: f"[MOCK LLM] {summarize(prompt, 200)}")
    query_results = [evaluate_query(store, agent, item) for item in BENCHMARK_QUERIES]
    relevant_count = sum(1 for result in query_results if result["relevant_in_top3"])

    return {
        "strategy": strategy_name,
        "chunk_count": store.get_collection_size(),
        "relevant_top3": relevant_count,
        "queries": query_results,
    }


def run_similarity_predictions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in SIMILARITY_PAIRS:
        vec_a = _mock_embed(pair["a"])
        vec_b = _mock_embed(pair["b"])
        score = round(compute_similarity(vec_a, vec_b), 4)
        actual = "high" if score >= 0.0 else "low"
        rows.append(
            {
                "sentence_a": pair["a"],
                "sentence_b": pair["b"],
                "prediction": pair["prediction"],
                "actual_score": score,
                "actual_label": actual,
                "correct": pair["prediction"] == actual,
            }
        )
    return rows


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 benchmark for Day 7 lab.")
    parser.add_argument("--export", type=str, default="", help="Optional JSON export path.")
    args = parser.parse_args()

    source_docs = load_source_documents()

    print_section("PHASE 2 — DOCUMENT INVENTORY")
    for index, doc in enumerate(source_docs, start=1):
        print(
            f"{index}. {doc.metadata['source']} | "
            f"{len(doc.content)} chars | "
            f"category={doc.metadata['category']} | "
            f"language={doc.metadata['language']}"
        )

    print_section("PHASE 2 — BASELINE CHUNKING (ChunkingStrategyComparator)")
    baseline = run_baseline_analysis()
    for filename, strategies in baseline.items():
        print(f"\n{filename}")
        for strategy, stats in strategies.items():
            print(f"  - {strategy}: count={stats['count']}, avg_length={stats['avg_length']}")

    print_section("PHASE 2 — SIMILARITY PREDICTIONS")
    similarity_rows = run_similarity_predictions()
    for index, row in enumerate(similarity_rows, start=1):
        print(
            f"{index}. pred={row['prediction']} | actual={row['actual_score']} ({row['actual_label']}) | "
            f"correct={row['correct']}"
        )
        print(f"   A: {row['sentence_a']}")
        print(f"   B: {row['sentence_b']}")

    print_section("PHASE 2 — RETRIEVAL BENCHMARK BY STRATEGY")
    strategy_results: dict[str, Any] = {}
    for strategy_name in STRATEGIES:
        result = run_strategy_benchmark(strategy_name, source_docs)
        strategy_results[strategy_name] = result
        print(
            f"\n{strategy_name}: chunks={result['chunk_count']} | "
            f"relevant_top3={result['relevant_top3']}/5"
        )
        for query in result["queries"]:
            mark = "OK" if query["relevant_in_top3"] else "MISS"
            print(
                f"  [{mark}] Q{query['id']}: {query['query'][:60]}... | "
                f"top1={query['top1_preview'][:70]}... | score={query['top1_score']}"
            )

    best_strategy = max(strategy_results.values(), key=lambda item: item["relevant_top3"])
    my_strategy = strategy_results["recursive_300"]

    payload = {
        "document_catalog": DOCUMENT_CATALOG,
        "baseline": baseline,
        "similarity_predictions": similarity_rows,
        "strategy_results": strategy_results,
        "best_strategy": best_strategy["strategy"],
        "my_strategy": "recursive_300",
        "benchmark_queries": BENCHMARK_QUERIES,
    }

    print_section("PHASE 2 — SUMMARY")
    print(f"Best retrieval strategy: {best_strategy['strategy']} ({best_strategy['relevant_top3']}/5)")
    print(f"My chosen strategy: recursive_300 ({my_strategy['relevant_top3']}/5)")

    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nExported JSON: {export_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
