# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Trọng Nguyên  
**Nhóm:** A3  
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector embedding hướng gần giống nhau trong không gian nhiều chiều — tức hai đoạn text có ngữ nghĩa tương tự, dù không dùng cùng từ ngữ.

**Ví dụ HIGH similarity:**
- Sentence A: Python is widely used for machine learning.
- Sentence B: Teams use Python for ML and data science.
- Tại sao tương đồng: Cùng chủ đề Python trong bối cảnh machine learning / data science.

**Ví dụ LOW similarity:**
- Sentence A: Python is a programming language for automation.
- Sentence B: Chocolate cake recipes need flour and eggs.
- Tại sao khác: Khác hoàn toàn về chủ đề (lập trình vs nấu ăn).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine đo góc giữa hai vector, ít bị ảnh hưởng bởi độ dài vector. Với embedding đã normalize, ta quan tâm hướng ngữ nghĩa hơn khoảng cách tuyệt đối — phù hợp hơn cho so sánh text.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23`  
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `ceil((10000 - 100) / (500 - 100)) = 25` — tăng từ 23 lên **25 chunks**. Overlap nhiều hơn giúp giữ ngữ cảnh ở ranh giới giữa các chunk, tránh mất ý khi retrieval chỉ trả về một đoạn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI Knowledge Assistant — RAG pipeline, vector store, chunking & customer support

**Tại sao nhóm chọn domain này?**
> Bộ tài liệu trong `data/` mô phỏng đúng use case của lab: thiết kế RAG, ghi chú vector store, thí nghiệm chunking, playbook support và ghi chú retrieval tiếng Việt. Domain cho phép benchmark đa dạng (kỹ thuật, support, song ngữ) và kiểm tra metadata filter `language=vi`.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | customer_support_playbook.txt | Lab sample data | 1692 | category=support, language=en, doc_type=playbook, audience=internal |
| 2 | python_intro.txt | Lab sample data | 1944 | category=technical, language=en, doc_type=tutorial, audience=public |
| 3 | vector_store_notes.md | Lab sample data | 2123 | category=technical, language=en, doc_type=notes, audience=public |
| 4 | rag_system_design.md | Lab sample data | 2391 | category=technical, language=en, doc_type=design, audience=internal |
| 5 | chunking_experiment_report.md | Lab sample data | 1987 | category=technical, language=en, doc_type=report, audience=internal |
| 6 | vi_retrieval_notes.md | Lab sample data | 1667 | category=technical, language=vi, doc_type=notes, audience=internal |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | support, technical | Lọc câu hỏi theo chủ đề (support vs kỹ thuật) |
| `language` | string | en, vi | Tránh retrieve nhầm tài liệu khác ngôn ngữ |
| `doc_type` | string | playbook, design, report | Phân biệt loại tài liệu (FAQ, thiết kế, báo cáo) |
| `audience` | string | public, internal | Giới hạn phạm vi nội dung phù hợp người hỏi |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu (`chunk_size=200`):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| customer_support_playbook.txt | FixedSizeChunker (`fixed_size`) | 10 | 187.2 | Trung bình — có thể cắt giữa câu |
| customer_support_playbook.txt | SentenceChunker (`by_sentences`) | 4 | 421.0 | Tốt — giữ ranh giới câu |
| customer_support_playbook.txt | RecursiveChunker (`recursive`) | 14 | 119.1 | Tốt — ưu tiên đoạn/câu trước khi cắt nhỏ |
| chunking_experiment_report.md | FixedSizeChunker (`fixed_size`) | 11 | 198.8 | Trung bình |
| chunking_experiment_report.md | SentenceChunker (`by_sentences`) | 5 | 395.6 | Tốt — chunk dài, dễ vượt kích thước lý tưởng |
| chunking_experiment_report.md | RecursiveChunker (`recursive`) | 18 | 108.4 | Tốt — phù hợp markdown nhiều section |
| vi_retrieval_notes.md | FixedSizeChunker (`fixed_size`) | 10 | 184.7 | Trung bình |
| vi_retrieval_notes.md | SentenceChunker (`by_sentences`) | 5 | 331.6 | Tốt |
| vi_retrieval_notes.md | RecursiveChunker (`recursive`) | 13 | 126.3 | Tốt — giữ đoạn tiếng Việt mạch lạc |

### Strategy Của Tôi

**Loại:** RecursiveChunker (`chunk_size=300` cho benchmark, `chunk_size=512` trong product)

**Mô tả cách hoạt động:**
> RecursiveChunker thử lần lượt các separator `["\n\n", "\n", ". ", " ", ""]`. Nếu đoạn vẫn dài hơn `chunk_size`, nó đệ quy xuống separator nhỏ hơn; nếu hết separator thì fallback sang cắt cứng theo ký tự. Các mảnh nhỏ liền kề được merge lại đến khi gần đạt `chunk_size`.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu nhóm gồm `.md` có heading/đoạn và `.txt` nhiều câu. Recursive chunking khai thác cấu trúc đoạn trước khi cắt nhỏ, phù hợp kết luận trong `chunking_experiment_report.md` và cho retrieval chính xác hơn trên câu hỏi kiến trúc (Q2) và pipeline (Q5).

**Code snippet (nếu custom):**
```python
chunker = RecursiveChunker(chunk_size=512)
chunks = chunker.chunk(document_text)
```

### So Sánh: Strategy của tôi vs Baseline

Benchmark retrieval trên toàn bộ corpus (6 docs) với embedder `all-MiniLM-L6-v2`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Toàn bộ corpus | fixed_size_300 (best baseline) | 49 | ~200 | 5/5 relevant top-3 |
| Toàn bộ corpus | **recursive_300 (của tôi)** | 71 | ~126 | **5/5 relevant top-3** |

> Cả hai strategy đều đạt 5/5 trên benchmark. Recursive có top-1 score cao hơn trên Q2 (0.600 vs 0.402) và Q5 (0.868 vs 0.752).

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker (300) | 10/10 | Chunk coherent, top-1 chính xác trên mọi query | Nhiều chunk hơn fixed_size |
| Nguyễn Thành Tài | FixedSizeChunker (300) | 10/10 | Ít chunk, kích thước ổn định | Có thể cắt giữa câu |
| Ngô Thị Ánh | SentenceChunker (2 câu) | 10/10 | Giữ câu trọn vẹn | Chunk dài, khó pinpoint đoạn ngắn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với embedder thật (MiniLM), cả 3 strategy đều 10/10. **RecursiveChunker** vẫn là lựa chọn tốt nhất cho production vì top-1 retrieval chất lượng cao nhất trên câu hỏi kiến trúc và pipeline, đồng thời phù hợp tài liệu markdown/đa ngôn ngữ trong domain.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách câu sau dấu chấm/chấm hỏi/chấm than, giữ punctuation trong câu. Gom `max_sentences_per_chunk` câu liên tiếp thành một chunk. Edge case: text rỗng trả về `[]`; nếu không tách được câu thì trả về nguyên đoạn text.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `_split` đệ quy theo danh sách separator. Base case: text rỗng → `[]`; text ≤ `chunk_size` → `[text]`; hết separator → fallback `FixedSizeChunker`. Sau khi split, merge các mảnh nhỏ liền kề nếu tổng độ dài không vượt `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được embed qua `embedding_fn`, lưu record gồm vector + metadata + content in-memory. Search embed query, tính cosine similarity với mọi record (vector đã normalize), sắp xếp giảm dần và trả về top-k.

**`search_with_filter` + `delete_document`** — approach:
> Filter **trước** khi search: chỉ giữ record có metadata khớp điều kiện (ví dụ `language=vi`), rồi mới tính similarity. Delete loại mọi record có `metadata['doc_id']` hoặc `parent_doc_id` trùng `doc_id` cần xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> `store.search(question, top_k)` lấy chunks liên quan → ghép context với citation `[n] (source, page)` → build prompt yêu cầu trả lời chỉ từ context và cite nguồn → gọi `llm_fn(prompt)`. Product mở rộng dùng OpenRouter (`openai/gpt-4o-mini`).

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.0.3, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestSentenceChunker::test_empty_text_returns_empty_list PASSED
... (38 tests khác, tất cả PASSED)

============================= 42 passed in 0.06s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Embedder: `all-MiniLM-L6-v2` (ngưỡng high ≥ 0.45)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is widely used for machine learning and data analysis. | Teams use Python for ML workflows and data science tasks. | high | 0.707 | ✅ |
| 2 | Vector stores retrieve similar embeddings for semantic search. | A vector database ranks chunks by similarity to the query. | high | 0.598 | ✅ |
| 3 | Customer support articles should use specific troubleshooting steps. | The billing API deployment requires Kubernetes credentials. | low | 0.141 | ✅ |
| 4 | Chunking splits documents into smaller retrieval units. | Recursive chunking tries paragraph boundaries before smaller splits. | high | 0.605 | ✅ |
| 5 | Python is a programming language used for automation. | Chocolate cake recipes need flour, eggs, and sugar. | low | -0.074 | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 5 có score âm (-0.074) — hai câu hoàn toàn khác chủ đề nhưng embedding vẫn có thể nằm gần nhau trong không gian vector. Điều này cho thấy embedding biểu diễn **hướng ngữ nghĩa** (cosine), không phải khớp từ khóa; cặp cùng chủ đề (pair 1, 2, 4) đều > 0.6 dù không trùng từ.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries trên implementation cá nhân: **RecursiveChunker(300)** + **all-MiniLM-L6-v2**.  
Lệnh: `python phase2_benchmark.py --embedder local --export report/phase2_results.json`

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What should support articles avoid writing? | Support articles should avoid vague statements such as 'check the settings' or 'contact engineering if needed.' They should specify the exact page, button, or log source. |
| 2 | What is the proposed architecture for the RAG system? | Ingestion chunks documents and stores segments with metadata; retrieval embeds questions and applies optional metadata filters; the application layer builds a prompt from top retrieved chunks. |
| 3 | Which chunking strategy performed best in the experiment? | Recursive chunking offered the best balance: it preserved context while staying within the target size range. |
| 4 | Metadata giúp retrieval tránh nhầm tài liệu như thế nào? | Metadata (phòng ban, ngôn ngữ, độ nhạy cảm, ngày cập nhật) giúp lọc tài liệu phù hợp, ví dụ tránh lấy marketing hoặc tài liệu tiếng Anh khi hỏi về tài liệu kỹ thuật tiếng Việt. |
| 5 | What are the four stages of a vector search pipeline? | Chunk documents, embed each chunk, store the vector and metadata, then embed the query and rank stored vectors by similarity. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What should support articles avoid writing? | "When writing support content, authors should avoid vague statements such as check the settings..." | 0.517 | ✅ | Mock LLM trả lời dựa trên customer_support_playbook |
| 2 | What is the proposed architecture for the RAG system? | "# RAG System Design for an Internal Knowledge Assistant ## Background" | 0.600 | ✅ | Mock LLM trả lời dựa trên rag_system_design.md |
| 3 | Which chunking strategy performed best in the experiment? | "Recursive chunking offered the best balance in the experiment..." | 0.717 | ✅ | Mock LLM trả lời dựa trên chunking_experiment_report.md |
| 4 | Metadata giúp retrieval tránh nhầm tài liệu như thế nào? | "Metadata cũng rất quan trọng. Ví dụ, một công ty có thể gắn nhãn tài liệu theo phòng ban, ngôn ngữ..." (filter `language=vi`) | 0.608 | ✅ | Mock LLM trả lời dựa trên vi_retrieval_notes.md |
| 5 | What are the four stages of a vector search pipeline? | "## Typical Workflow A common vector search pipeline has four stages:" | 0.868 | ✅ | Mock LLM trả lời dựa trên vector_store_notes.md |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **5 / 5**

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> So sánh 3 chunking strategy trong nhóm cho thấy fixed_size tiết kiệm chunk count nhất, sentence giữ câu trọn vẹn tốt, còn recursive cân bằng giữa kích thước và ngữ cảnh — mỗi strategy có trade-off rõ ràng tùy loại tài liệu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Metadata filter (`language=vi`) là điểm then chốt để retrieval đa ngôn ngữ không bị nhiễu; và việc dùng embedder thật (MiniLM) thay vì mock embedder thay đổi hoàn toàn kết quả benchmark từ 3/5 lên 5/5.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> (1) Chạy benchmark sớm với embedder thật thay vì mock. (2) Thêm metadata `category` filter cho câu hỏi support vs technical. (3) Chuẩn hóa kích thước tài liệu upload để không làm nhiễu retrieval. (4) Lưu vector persistently (ChromaDB) thay vì in-memory cho use case production.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **87 / 100** |
