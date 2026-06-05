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
import os
from typing import Any, Callable

from dotenv import load_dotenv

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
UPLOAD_DIR = DATA_DIR / "uploads"

load_dotenv(dotenv_path=ROOT / ".env", override=False)

DOCUMENT_CATALOG: list[dict[str, Any]] = [
    {
        "path": "paper_I_random_matrix.txt",
        "category": "statistics",
        "language": "en",
        "doc_type": "preprint",
        "topic": "random_matrix",
    },
    {
        "path": "paper_II_weak_iv_estimators.txt",
        "category": "econometrics",
        "language": "en",
        "doc_type": "preprint",
        "topic": "instrumental_variables",
    },
    {
        "path": "paper_III_sir_seir_identifiability.txt",
        "category": "epidemiology",
        "language": "en",
        "doc_type": "preprint",
        "topic": "compartmental_models",
    },
    {
        "path": "paper_IV_krylov_preconditioners.txt",
        "category": "numerical_methods",
        "language": "en",
        "doc_type": "preprint",
        "topic": "sparse_linear_systems",
    },
    {
        "path": "paper_V_gard_lora.txt",
        "category": "machine_learning",
        "language": "en",
        "doc_type": "preprint",
        "topic": "parameter_efficient_finetuning",
    },
    {
        "path": "paper_VI_lace_exploration.txt",
        "category": "reinforcement_learning",
        "language": "en",
        "doc_type": "preprint",
        "topic": "exploration",
    },
]

BASELINE_DOCS = [
    "paper_I_random_matrix.txt",
    "paper_III_sir_seir_identifiability.txt",
    "paper_V_gard_lora.txt",
]

BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {
        "id": 1,
        "query": "What is MPCX and what finite-dimensional correction problem does it study?",
        "gold_answer": (
            "MPCX is a computational framework that compares finite-dimensional corrections "
            "to the Marchenko-Pastur law for Gaussian Wishart matrices, evaluating bulk-sensitive "
            "and edge-sensitive strategies for dimensions from 50 to 5000."
        ),
        "expected_source": "paper_I_random_matrix",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Which IV estimators formed the lowest-loss cluster in the IVX benchmark?",
        "gold_answer": (
            "LIML, Fuller(1), and Fuller(4) formed a tightly clustered low-loss group "
            "(metrics around 0.84), while 2SLS and JIVE had much larger losses."
        ),
        "expected_source": "paper_II_weak_iv_estimators",
        "filter": None,
    },
    {
        "id": 3,
        "query": "What is PRIM and what identifiability diagnostics does it compare for epidemic models?",
        "gold_answer": (
            "PRIM benchmarks structural and practical identifiability for SIR and SEIR models "
            "using profile likelihood, Fisher Information diagnostics, and maximum-likelihood "
            "estimation on synthetic outbreaks."
        ),
        "expected_source": "paper_III_sir_seir_identifiability",
        "filter": None,
    },
    {
        "id": 4,
        "query": "How does GARD allocate LoRA ranks using gradient spectral analysis?",
        "gold_answer": (
            "GARD estimates per-layer gradient covariance spectra, derives an effective "
            "dimensionality signal, and maps it to discrete LoRA ranks under a fixed global "
            "adapter parameter budget."
        ),
        "expected_source": "paper_V_gard_lora",
        "filter": {"category": "machine_learning"},
    },
    {
        "id": 5,
        "query": "Which Krylov preconditioners were compared on sparse linear systems?",
        "gold_answer": (
            "The study compares Jacobi, SSOR, ILU, and AMG preconditioners with Krylov solvers "
            "such as CG, GMRES, and BiCGSTAB on structured and unstructured sparse systems."
        ),
        "expected_source": "paper_IV_krylov_preconditioners",
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


def resolve_embedder(name: str):
    selected = (name or os.getenv("PHASE2_EMBEDDER", "local")).strip().lower()
    if selected == "mock":
        return _mock_embed, "mock"

    try:
        from src.embeddings import LocalEmbedder

        embedder = LocalEmbedder()
        backend = getattr(embedder, "_backend_name", "all-MiniLM-L6-v2")
        return embedder, backend
    except Exception as exc:
        print(f"[warn] Local embedder unavailable ({exc}); falling back to mock.")
        return _mock_embed, "mock"


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
                    "topic": entry.get("topic", entry["category"]),
                },
            )
        )

    if UPLOAD_DIR.exists():
        for path in sorted(UPLOAD_DIR.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            if len(content.strip()) < 200:
                continue
            documents.append(
                Document(
                    id=path.stem,
                    content=content,
                    metadata={
                        "source": str(path.relative_to(ROOT)),
                        "category": "upload",
                        "language": "vi",
                        "doc_type": "notes",
                        "audience": "internal",
                        "uploaded": True,
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


def run_strategy_benchmark(
    strategy_name: str,
    source_docs: list[Document],
    embedder: Any,
) -> dict[str, Any]:
    chunker = STRATEGIES[strategy_name]()
    store = EmbeddingStore(collection_name=f"phase2_{strategy_name}", embedding_fn=embedder)
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


def run_similarity_predictions(embedder: Any, embedder_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    high_threshold = 0.45 if embedder_name != "mock" else 0.0
    for pair in SIMILARITY_PAIRS:
        vec_a = embedder(pair["a"])
        vec_b = embedder(pair["b"])
        score = round(compute_similarity(vec_a, vec_b), 4)
        actual = "high" if score >= high_threshold else "low"
        rows.append(
            {
                "embedder": embedder_name,
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


def format_text_report(payload: dict[str, Any]) -> str:
    """Plain-text report for report/phase2_output.txt (UTF-8)."""
    lines: list[str] = []
    embedder = payload["embedder"]

    lines.append("=" * 72)
    lines.append("PHASE 2 — DOCUMENT INVENTORY")
    lines.append("=" * 72)
    for index, doc in enumerate(payload["document_inventory"], start=1):
        lines.append(
            f"{index}. {doc['source']} | {doc['chars']} chars | "
            f"category={doc['category']} | language={doc['language']}"
        )

    lines.append("")
    lines.append("=" * 72)
    lines.append("PHASE 2 — BASELINE CHUNKING (ChunkingStrategyComparator)")
    lines.append("=" * 72)
    for filename, strategies in payload["baseline"].items():
        lines.append(f"\n{filename}")
        for strategy, stats in strategies.items():
            lines.append(
                f"  - {strategy}: count={stats['count']}, avg_length={stats['avg_length']}"
            )

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"PHASE 2 — SIMILARITY PREDICTIONS ({embedder})")
    lines.append("=" * 72)
    for index, row in enumerate(payload["similarity_predictions"], start=1):
        lines.append(
            f"{index}. pred={row['prediction']} | actual={row['actual_score']} "
            f"({row['actual_label']}) | correct={row['correct']}"
        )
        lines.append(f"   A: {row['sentence_a']}")
        lines.append(f"   B: {row['sentence_b']}")

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"PHASE 2 — RETRIEVAL BENCHMARK BY STRATEGY ({embedder})")
    lines.append("=" * 72)
    for strategy_name, result in payload["strategy_results"].items():
        lines.append(
            f"\n{strategy_name}: chunks={result['chunk_count']} | "
            f"relevant_top3={result['relevant_top3']}/5"
        )
        for query in result["queries"]:
            mark = "OK" if query["relevant_in_top3"] else "MISS"
            lines.append(
                f"  [{mark}] Q{query['id']}: {query['query'][:60]}... | "
                f"top1={query['top1_preview'][:70]}... | score={query['top1_score']}"
            )

    my_strategy = payload["strategy_results"][payload["my_strategy"]]
    lines.append("")
    lines.append("=" * 72)
    lines.append("PHASE 2 — SUMMARY")
    lines.append("=" * 72)
    lines.append(
        f"Best retrieval strategy: {payload['best_strategy']} "
        f"({payload['strategy_results'][payload['best_strategy']]['relevant_top3']}/5)"
    )
    lines.append(
        f"My chosen strategy: {payload['my_strategy']} ({my_strategy['relevant_top3']}/5)"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 benchmark for Day 7 lab.")
    parser.add_argument("--export", type=str, default="", help="Optional JSON export path.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional UTF-8 text report path (e.g. report/phase2_output.txt).",
    )
    parser.add_argument(
        "--embedder",
        choices=["local", "mock"],
        default=os.getenv("PHASE2_EMBEDDER", "local"),
        help="Embedding backend for retrieval benchmark (default: local MiniLM).",
    )
    parser.add_argument(
        "--my-strategy",
        choices=list(STRATEGIES.keys()),
        default=os.getenv("PHASE2_MY_STRATEGY", "fixed_size_300"),
        help="Personal chunking strategy for report export (default: fixed_size_300).",
    )
    args = parser.parse_args()

    embedder, embedder_name = resolve_embedder(args.embedder)
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

    print_section(f"PHASE 2 — SIMILARITY PREDICTIONS ({embedder_name})")
    similarity_rows = run_similarity_predictions(embedder, embedder_name)
    for index, row in enumerate(similarity_rows, start=1):
        print(
            f"{index}. pred={row['prediction']} | actual={row['actual_score']} ({row['actual_label']}) | "
            f"correct={row['correct']}"
        )
        print(f"   A: {row['sentence_a']}")
        print(f"   B: {row['sentence_b']}")

    print_section(f"PHASE 2 — RETRIEVAL BENCHMARK BY STRATEGY ({embedder_name})")
    strategy_results: dict[str, Any] = {}
    for strategy_name in STRATEGIES:
        result = run_strategy_benchmark(strategy_name, source_docs, embedder)
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
    my_strategy = strategy_results[args.my_strategy]

    payload = {
        "embedder": embedder_name,
        "document_count": len(source_docs),
        "document_inventory": [
            {
                "source": doc.metadata["source"],
                "chars": len(doc.content),
                "category": doc.metadata["category"],
                "language": doc.metadata["language"],
            }
            for doc in source_docs
        ],
        "document_catalog": DOCUMENT_CATALOG,
        "baseline": baseline,
        "similarity_predictions": similarity_rows,
        "strategy_results": strategy_results,
        "best_strategy": best_strategy["strategy"],
        "my_strategy": args.my_strategy,
        "benchmark_queries": BENCHMARK_QUERIES,
    }

    print_section("PHASE 2 — SUMMARY")
    print(f"Best retrieval strategy: {best_strategy['strategy']} ({best_strategy['relevant_top3']}/5)")
    print(f"My chosen strategy: {args.my_strategy} ({my_strategy['relevant_top3']}/5)")

    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nExported JSON: {export_path}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(format_text_report(payload), encoding="utf-8")
        print(f"Exported text report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
