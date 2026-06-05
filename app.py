"""Streamlit UI for Day 7 — Embedding & Vector Store lab."""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from src.agent import KnowledgeBaseAgent
from src.store import EmbeddingStore
from ui.helpers import (
    benchmark_results_to_csv,
    compare_chunking_strategies,
    create_embedder,
    create_mock_llm,
    create_openai_llm,
    get_default_embedding_provider,
    get_embedder_backend_name,
    has_openai_api_key,
    list_data_files,
    load_documents_from_paths,
    load_dotenv_config,
)

st.set_page_config(
    page_title="Day 7 — Embedding & Vector Store",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .main-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }

    .metric-card strong {
        color: #0f172a;
    }

    .empty-state {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: #64748b;
    }

    .chunk-preview {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.5;
    }

    .answer-box {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #312e81;
        line-height: 1.6;
    }

    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 500;
    }
</style>
"""

STRATEGY_LABELS = {
    "fixed_size": "Fixed size",
    "by_sentences": "By sentences",
    "recursive": "Recursive",
}


def init_session_state() -> None:
    defaults = {
        "store": None,
        "indexed_doc_ids": set(),
        "store_provider": None,
        "benchmark_results": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_or_create_store(provider: str) -> EmbeddingStore:
    if (
        st.session_state.store is None
        or st.session_state.store_provider != provider
    ):
        embedder = create_embedder(provider)
        st.session_state.store = EmbeddingStore(
            collection_name="streamlit_ui_store",
            embedding_fn=embedder,
        )
        st.session_state.store_provider = provider
        st.session_state.indexed_doc_ids = set()
        st.session_state.benchmark_results = []
    return st.session_state.store


def render_sidebar() -> tuple[str, int, str]:
    st.sidebar.markdown("### Cấu hình")
    load_dotenv_config()

    provider_options = ["mock", "local", "openai"]
    default_provider = get_default_embedding_provider()
    if default_provider not in provider_options:
        default_provider = "mock"

    provider = st.sidebar.selectbox(
        "Embedding provider",
        provider_options,
        index=provider_options.index(default_provider),
        help="mock = mặc định lab, local = sentence-transformers, openai = API",
    )

    top_k = st.sidebar.slider("Top-k retrieval", min_value=1, max_value=10, value=3)

    llm_options = ["mock"]
    if has_openai_api_key():
        llm_options.append("openai")
    llm_provider = st.sidebar.selectbox("LLM provider (Tab Ask)", llm_options)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Trạng thái index**")
    store = st.session_state.store
    if store is None:
        st.sidebar.info("Chưa có vector store. Hãy index tài liệu ở Tab Documents.")
    else:
        st.sidebar.success(f"Đã index: {len(st.session_state.indexed_doc_ids)} tài liệu")
        st.sidebar.caption(f"Chunks trong store: {store.get_collection_size()}")
        backend = get_embedder_backend_name(store._embedding_fn)
        st.sidebar.caption(f"Backend: {backend}")

    return provider, top_k, llm_provider


def render_documents_tab(provider: str) -> None:
    st.markdown('<p class="main-header">Documents</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Chọn file trong thư mục <code>data/</code> và index vào vector store.</p>',
        unsafe_allow_html=True,
    )

    data_files = list_data_files()
    if not data_files:
        st.markdown(
            '<div class="empty-state">Chưa có file .txt hoặc .md trong thư mục <code>data/</code>.</div>',
            unsafe_allow_html=True,
        )
        return

    file_labels = [str(path.relative_to(path.parent.parent)) for path in data_files]
    selected_labels = st.multiselect(
        "Chọn tài liệu để index",
        file_labels,
        default=file_labels,
    )

    selected_paths = [
        path
        for path, label in zip(data_files, file_labels)
        if label in selected_labels
    ]

    with st.expander("Xem metadata từng file", expanded=False):
        preview_docs = load_documents_from_paths(selected_paths)
        if preview_docs:
            rows = [
                {
                    "id": doc.id,
                    "source": doc.metadata.get("source"),
                    "chars": doc.metadata.get("chars"),
                    "extension": doc.metadata.get("extension"),
                }
                for doc in preview_docs
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Chưa chọn file hợp lệ.")

    col1, col2 = st.columns([1, 3])
    with col1:
        index_clicked = st.button("Index documents", type="primary", use_container_width=True)

    store = get_or_create_store(provider)
    backend_name = get_embedder_backend_name(store._embedding_fn)

    if index_clicked:
        if not selected_paths:
            st.warning("Hãy chọn ít nhất một file.")
        else:
            with st.spinner("Đang embed và index tài liệu..."):
                docs = load_documents_from_paths(selected_paths)
                new_docs = [doc for doc in docs if doc.id not in st.session_state.indexed_doc_ids]
                if new_docs:
                    store.add_documents(new_docs)
                    for doc in new_docs:
                        st.session_state.indexed_doc_ids.add(doc.id)
                st.success(f"Đã index {len(new_docs)} tài liệu mới.")

    st.markdown(
        f"""
        <div class="metric-card">
            <strong>Embedding backend:</strong> {backend_name}<br>
            <strong>Tài liệu đã index:</strong> {len(st.session_state.indexed_doc_ids)}<br>
            <strong>Tổng chunks:</strong> {store.get_collection_size()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.indexed_doc_ids:
        indexed_docs = load_documents_from_paths(
            [get_data_path(doc_id) for doc_id in sorted(st.session_state.indexed_doc_ids)]
        )
        st.dataframe(
            [
                {
                    "doc_id": doc.id,
                    "source": doc.metadata.get("source"),
                    "chars": doc.metadata.get("chars"),
                }
                for doc in indexed_docs
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown(
            '<div class="empty-state">Chưa index tài liệu nào. Chọn file và nhấn <strong>Index documents</strong>.</div>',
            unsafe_allow_html=True,
        )


def get_data_path(doc_id: str) -> Path:
    data_dir = Path(__file__).resolve().parent / "data"
    for ext in (".txt", ".md"):
        candidate = data_dir / f"{doc_id}{ext}"
        if candidate.exists():
            return candidate
    return data_dir / doc_id


def render_ask_tab(top_k: int, llm_provider: str) -> None:
    st.markdown('<p class="main-header">Ask — RAG Chat</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Hỏi đáp dựa trên vector store đã index. Xem chunks retrieve và câu trả lời agent.</p>',
        unsafe_allow_html=True,
    )

    store = st.session_state.store
    if store is None or store.get_collection_size() == 0:
        st.markdown(
            '<div class="empty-state">Chưa có dữ liệu trong store. Vào tab <strong>Documents</strong> để index trước.</div>',
            unsafe_allow_html=True,
        )
        return

    question = st.text_area(
        "Câu hỏi",
        placeholder="Ví dụ: Python được dùng cho những mục đích gì?",
        height=100,
    )

    ask_clicked = st.button("Search / Ask", type="primary")

    if not ask_clicked:
        return
    if not question.strip():
        st.warning("Hãy nhập câu hỏi.")
        return

    with st.spinner("Đang retrieve và tạo câu trả lời..."):
        results = store.search(question.strip(), top_k=top_k)

        llm_fn = create_mock_llm()
        if llm_provider == "openai":
            openai_llm = create_openai_llm()
            if openai_llm is not None:
                llm_fn = openai_llm
            else:
                st.info("Không dùng được OpenAI chat — fallback về mock LLM.")

        agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
        answer = agent.answer(question.strip(), top_k=top_k)

    st.markdown("#### Retrieved chunks")
    if not results:
        st.info("Không tìm thấy chunk phù hợp.")
    else:
        for index, result in enumerate(results, start=1):
            source = result["metadata"].get("source", "unknown")
            preview = html.escape(result["content"][:280].replace("\n", " "))
            safe_source = html.escape(str(source))
            st.markdown(
                f"""
                <div class="chunk-preview">
                    <strong>#{index}</strong> · score <strong>{result['score']:.4f}</strong> · {safe_source}<br>
                    {preview}{"..." if len(result["content"]) > 280 else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Câu trả lời")
    st.markdown(
        f'<div class="answer-box">{html.escape(answer)}</div>',
        unsafe_allow_html=True,
    )


def render_chunking_tab() -> None:
    st.markdown('<p class="main-header">Chunking Lab</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">So sánh 3 chiến lược chunking trên một file mẫu.</p>',
        unsafe_allow_html=True,
    )

    data_files = list_data_files()
    if not data_files:
        st.markdown(
            '<div class="empty-state">Không có file trong <code>data/</code> để thử nghiệm.</div>',
            unsafe_allow_html=True,
        )
        return

    file_labels = [path.name for path in data_files]
    selected_name = st.selectbox("Chọn file", file_labels)
    selected_path = data_files[file_labels.index(selected_name)]

    col1, col2, col3 = st.columns(3)
    with col1:
        chunk_size = st.number_input("Chunk size", min_value=50, max_value=2000, value=200, step=50)
    with col2:
        overlap = st.number_input("Overlap (fixed)", min_value=0, max_value=500, value=0, step=10)
    with col3:
        max_sentences = st.number_input("Max sentences", min_value=1, max_value=20, value=3, step=1)

    compare_clicked = st.button("Compare strategies", type="primary")

    if not compare_clicked:
        return

    text = selected_path.read_text(encoding="utf-8")
    with st.spinner("Đang so sánh chunking strategies..."):
        comparison = compare_chunking_strategies(
            text=text,
            chunk_size=int(chunk_size),
            overlap=int(overlap),
            max_sentences=int(max_sentences),
        )

    table_rows = [
        {
            "strategy": STRATEGY_LABELS.get(name, name),
            "count": stats["count"],
            "avg_length": round(stats["avg_length"], 1),
        }
        for name, stats in comparison.items()
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    preview_strategy = st.selectbox(
        "Xem preview chunks",
        list(comparison.keys()),
        format_func=lambda key: STRATEGY_LABELS.get(key, key),
    )
    preview_chunks = comparison[preview_strategy]["chunks"][:5]
    st.markdown(f"**Preview — {STRATEGY_LABELS.get(preview_strategy, preview_strategy)}** (5 chunk đầu)")
    for index, chunk in enumerate(preview_chunks, start=1):
        preview = html.escape(chunk[:350].replace("\n", " "))
        st.markdown(
            f"""
            <div class="chunk-preview">
                <strong>Chunk {index}</strong> ({len(chunk)} ký tự)<br>
                {preview}{"..." if len(chunk) > 350 else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_benchmark_tab(top_k: int) -> None:
    st.markdown('<p class="main-header">Benchmark (Phase 2)</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Chạy batch 5 benchmark queries và đánh giá retrieval thủ công.</p>',
        unsafe_allow_html=True,
    )

    store = st.session_state.store
    if store is None or store.get_collection_size() == 0:
        st.markdown(
            '<div class="empty-state">Index tài liệu trước khi chạy benchmark.</div>',
            unsafe_allow_html=True,
        )
        return

    default_queries = "\n".join(
        [
            "Python được dùng cho những mục đích gì?",
            "Vector store lưu trữ dữ liệu như thế nào?",
            "RAG hoạt động theo các bước nào?",
            "Chunking strategy nào phù hợp cho tài liệu dài?",
            "Cosine similarity khác Euclidean distance thế nào?",
        ]
    )
    queries_text = st.text_area("Benchmark queries (mỗi dòng 1 query)", value=default_queries, height=160)

    col1, col2 = st.columns(2)
    with col1:
        filter_category = st.text_input("Metadata filter — category (tùy chọn)", value="")
    with col2:
        filter_language = st.text_input("Metadata filter — language (tùy chọn)", value="")

    run_clicked = st.button("Run benchmark", type="primary")

    if run_clicked:
        queries = [line.strip() for line in queries_text.splitlines() if line.strip()]
        if not queries:
            st.warning("Hãy nhập ít nhất một query.")
            return

        metadata_filter: dict[str, str] = {}
        if filter_category.strip():
            metadata_filter["category"] = filter_category.strip()
        if filter_language.strip():
            metadata_filter["language"] = filter_language.strip()

        with st.spinner("Đang chạy benchmark..."):
            results = []
            for query in queries:
                if metadata_filter:
                    hits = store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
                else:
                    hits = store.search(query, top_k=top_k)

                top1 = hits[0] if hits else None
                results.append(
                    {
                        "query": query,
                        "top1_score": round(top1["score"], 4) if top1 else None,
                        "top1_source": top1["metadata"].get("source") if top1 else None,
                        "top1_preview": (top1["content"][:200] if top1 else ""),
                        "relevant": False,
                    }
                )
            st.session_state.benchmark_results = results

    results = st.session_state.benchmark_results
    if not results:
        st.info("Nhấn **Run benchmark** để tạo bảng kết quả.")
        return

    for index, row in enumerate(results):
        cols = st.columns([3, 1, 2, 1])
        with cols[0]:
            st.markdown(f"**Q{index + 1}:** {row['query']}")
            if row.get("top1_preview"):
                st.caption(row["top1_preview"])
        with cols[1]:
            score = row.get("top1_score")
            st.metric("Score", f"{score:.4f}" if score is not None else "—")
        with cols[2]:
            st.caption(row.get("top1_source") or "—")
        with cols[3]:
            results[index]["relevant"] = st.checkbox(
                "Relevant",
                value=row.get("relevant", False),
                key=f"bench_relevant_{index}",
            )

    st.session_state.benchmark_results = results

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "Export JSON",
            data=json.dumps(results, ensure_ascii=False, indent=2),
            file_name="benchmark_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with export_col2:
        st.download_button(
            "Export CSV",
            data=benchmark_results_to_csv(results),
            file_name="benchmark_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    st.markdown('<p class="main-header">Day 7 — Embedding & Vector Store</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Lab UI: index tài liệu, hỏi đáp RAG, thử chunking, chạy benchmark.</p>',
        unsafe_allow_html=True,
    )

    provider, top_k, llm_provider = render_sidebar()
    get_or_create_store(provider)

    tab_docs, tab_ask, tab_chunk, tab_bench = st.tabs(
        ["Documents", "Ask", "Chunking Lab", "Benchmark"]
    )

    with tab_docs:
        render_documents_tab(provider)
    with tab_ask:
        render_ask_tab(top_k, llm_provider)
    with tab_chunk:
        render_chunking_tab()
    with tab_bench:
        render_benchmark_tab(top_k)


if __name__ == "__main__":
    main()
