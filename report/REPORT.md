# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** Solo / AI Knowledge Assistant Lab
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector embedding hướng gần giống nhau trong không gian nhiều chiều — tức hai đoạn text có ngữ nghĩa tương tự, dù không trùng từ ngữ.

**Ví dụ HIGH similarity:**
- Sentence A: Python is widely used for machine learning.
- Sentence B: Teams use Python for ML and data science.
- Tại sao tương đồng: Cùng nói về Python trong bối cảnh ML/data.

**Ví dụ LOW similarity:**
- Sentence A: Python is a programming language for automation.
- Sentence B: Chocolate cake recipes need flour and eggs.
- Tại sao khác: Khác chủ đề hoàn toàn.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine đo góc giữa hai vector, ít bị ảnh hưởng bởi độ dài vector; phù hợp khi embedding đã normalize và ta quan tâm hướng ngữ nghĩa hơn độ lớn tuyệt đối.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25` — tăng từ 23 lên **25 chunks**. Overlap nhiều hơn giúp giữ ngữ cảnh ở ranh giới chunk, tránh mất ý khi retrieval chỉ lấy một đoạn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI Knowledge Assistant — RAG, Vector Store, Chunking & Customer Support

**Tại sao nhóm chọn domain này?**
> Bộ tài liệu mẫu trong `data/` mô phỏng đúng pipeline lab: thiết kế RAG, ghi chú vector store, thí nghiệm chunking, playbook support, và ghi chú retrieval tiếng Việt. Domain này cho phép benchmark đa dạng (kỹ thuật, support, song ngữ) mà không cần thu thập PDF ngoài.

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
| customer_support_playbook.txt | FixedSizeChunker (`fixed_size`) | 9 | 188.0 | Trung bình — có thể cắt giữa câu |
| customer_support_playbook.txt | SentenceChunker (`by_sentences`) | 4 | 418.2 | Tốt — giữ ranh giới câu, chunk hơi dài |
| customer_support_playbook.txt | RecursiveChunker (`recursive`) | 14 | 124.0 | Tốt — ưu tiên đoạn/câu trước khi cắt nhỏ |
| chunking_experiment_report.md | FixedSizeChunker (`fixed_size`) | 10 | 198.7 | Trung bình |
| chunking_experiment_report.md | SentenceChunker (`by_sentences`) | 5 | 392.6 | Tốt cho đọc, dễ vượt kích thước lý tưởng |
| chunking_experiment_report.md | RecursiveChunker (`recursive`) | 18 | 109.2 | Tốt — phù hợp markdown nhiều section |
| vi_retrieval_notes.md | FixedSizeChunker (`fixed_size`) | 9 | 185.2 | Trung bình |
| vi_retrieval_notes.md | SentenceChunker (`by_sentences`) | 5 | 328.8 | Tốt |
| vi_retrieval_notes.md | RecursiveChunker (`recursive`) | 13 | 127.2 | Tốt — giữ đoạn tiếng Việt mạch lạc |

### Strategy Của Tôi

**Loại:** RecursiveChunker (custom params: `chunk_size=300`)

**Mô tả cách hoạt động:**
> RecursiveChunker thử lần lượt separators `["\n\n", "\n", ". ", " ", ""]`. Nếu đoạn vẫn dài hơn 300 ký tự, nó đệ quy xuống separator nhỏ hơn; nếu hết separator thì cắt theo ký tự. Các mảnh nhỏ được merge lại đến khi gần đạt `chunk_size`.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu lab gồm `.md` có heading/đoạn và `.txt` nhiều câu. Recursive chunking khai thác cấu trúc đoạn trước, phù hợp với kết luận trong `chunking_experiment_report.md` — recursive là default tốt cho mixed technical docs.

**Code snippet (nếu custom):**
```python
# Strategy của tôi: dùng RecursiveChunker có sẵn với chunk_size tinh chỉnh
chunker = RecursiveChunker(chunk_size=300)
chunks = chunker.chunk(document_text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| customer_support_playbook.txt | best baseline (recursive @200) | 14 | 124.0 | 6/10 — coherent nhưng chunk nhỏ |
| customer_support_playbook.txt | **của tôi (recursive @300)** | ~10* | ~170* | 7/10 — ít chunk hơn, giữ thêm ngữ cảnh |
| Toàn bộ corpus (6 docs) | fixed_size_300 | 49 | — | **3/5** relevant top-3 |
| Toàn bộ corpus (6 docs) | sentence_2 | 49 | — | **2/5** relevant top-3 |
| Toàn bộ corpus (6 docs) | **recursive_300 (của tôi)** | 71 | — | **3/5** relevant top-3 |

\*Ước lượng từ cùng tỷ lệ scale khi tăng `chunk_size` 200→300.

### So Sánh Với Thành Viên Khác

*Làm solo — so sánh 3 strategy như 3 "thành viên ảo":*

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker (300) | 6/10 (3/5 top-3) | Chunk coherent, tốt với markdown & tiếng Việt | Nhiều chunk (71), Q1/Q2 miss với mock embedder |
| Strategy A | FixedSizeChunker (300, overlap 50) | 6/10 (3/5 top-3) | Ổn định kích thước, Q1 & Q3 tốt | Cắt giữa câu, Q5 miss |
| Strategy B | SentenceChunker (2 câu/chunk) | 4/10 (2/5 top-3) | Dễ đọc, Q5 có stage 4 đúng | Chunk dài, Q1/Q2/Q3 kém |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với mock embedder, **FixedSize 300** và **Recursive 300** hòa **3/5**. Về mặt thiết kế dữ liệu, **Recursive 300** vẫn là lựa chọn tốt hơn vì giữ cấu trúc đoạn/câu — phù hợp docs kỹ thuật mixed. FixedSize thắng nhẹ trên benchmark này vì mock embedding nhạy từ khóa hơn cấu trúc ngữ nghĩa.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?:\.\n|\.\s|!\s|\?\s)` để tách câu, strip whitespace, gom `max_sentences_per_chunk` câu thành một chunk. Edge case: text rỗng → `[]`; không tách được câu → trả về cả text đã strip.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Đệ quy theo danh sách separator; base case: text ≤ chunk_size hoặc hết separator thì cắt cứng theo ký tự. Merge các mảnh nhỏ liền kề nếu tổng độ dài không vượt `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document embed qua `embedding_fn`, lưu record in-memory (hoặc ChromaDB nếu có). Search embed query rồi xếp hạng bằng dot product với vector đã lưu.

**`search_with_filter` + `delete_document`** — approach:
> Filter **trước** khi search — lọc records theo metadata key-value khớp hoàn toàn, rồi mới tính similarity. Delete loại mọi record có `metadata['doc_id']` trùng.

### KnowledgeBaseAgent

**`answer`** — approach:
> `store.search(question, top_k)` → ghép context bằng `\n\n` → prompt có section Context + Question → gọi `llm_fn(prompt)`.

### Test Results

```
============================= test session starts =============================
collected 42 items
tests\test_solution.py ..........................................        [100%]
============================= 42 passed in 0.95s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*Dùng `_mock_embed` + `compute_similarity`. Ngưỡng actual: score ≥ 0 = high.*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is widely used for ML and data analysis. | Teams use Python for ML workflows and data science. | high | -0.0882 | Không |
| 2 | Vector stores retrieve similar embeddings. | A vector database ranks chunks by similarity. | high | -0.0150 | Không |
| 3 | Support articles should use specific steps. | Billing API deployment requires Kubernetes. | low | -0.0134 | Có |
| 4 | Chunking splits documents into retrieval units. | Recursive chunking tries paragraph boundaries first. | high | -0.0083 | Không |
| 5 | Python is used for automation. | Chocolate cake recipes need flour and eggs. | low | 0.0739 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp 1 và 5 bất ngờ nhất: câu cùng chủ đề Python+ML bị score âm, trong khi Python vs bánh có score dương nhẹ. Mock embedder dựa trên hash — **không capture ngữ nghĩa**. Điều này nhấn mạnh: similarity predictions chỉ có ý nghĩa khi dùng embedder thật (MiniLM/OpenAI); với mock chỉ nên test pipeline, không test semantic intuition.

---

## 6. Results — Cá nhân (10 điểm)

Chạy benchmark bằng `py phase2_benchmark.py` với strategy **RecursiveChunker(300)**.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What should support articles avoid writing? | Tránh câu mơ hồ ("check the settings"); phải ghi rõ page/button/log source. |
| 2 | What is the proposed architecture for the RAG system? | Ingestion chunk+metadata → retrieval embed+filter → app layer build prompt từ top chunks. |
| 3 | Which chunking strategy performed best in the experiment? | Recursive chunking — cân bằng context và kích thước. |
| 4 | Metadata giúp retrieval tránh nhầm tài liệu như thế nào? | Lọc theo phòng ban, ngôn ngữ, độ nhạy cảm, ngày cập nhật (filter `language=vi`). |
| 5 | What are the four stages of a vector search pipeline? | Chunk → embed chunk → store vector+metadata → embed query & rank. |

### Kết Quả Của Tôi (RecursiveChunker 300)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Support avoid writing? | Chunk từ chunking_experiment_report (recursive chunking...) | -0.478 | Không (miss top-3) | Mock LLM dựa trên chunk sai |
| 2 | RAG architecture? | Chunk từ chunking_experiment_report (no universal best strategy...) | -0.348 | Không (miss top-3) | Mock LLM — không có kiến trúc RAG |
| 3 | Best chunking strategy? | rag_system_design (failure cases...) — nhưng **chunking_report có trong top-3** | -0.382 | Có (top-3) | Top-1 chưa đúng, top-3 có doc đúng |
| 4 | Metadata tiếng Việt? | vi_retrieval_notes — chunk về chất lượng chunking & metadata | 0.177 | Có | Filter `language=vi` hiệu quả |
| 5 | Four stages pipeline? | vector_store_notes có trong top-3; top-1 là đoạn evaluation | -0.360 | Có (top-3) | Top-1 chưa phải 4 stages |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **3 / 5**

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> So sánh 3 strategy cho thấy không có "one size fits all": FixedSize thắng nhẹ trên mock embedder, SentenceChunker thua vì chunk quá dài gộp nhiều ý, Recursive cân bằng coherence tốt nhất cho docs markdown.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *(Điền sau buổi demo lớp)* — Dự kiến: metadata filter và benchmark queries cụ thể quan trọng hơn đổi chunk_size 200→300.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> (1) Tách riêng corpus support vs technical để giảm nhiễu cross-topic. (2) Thêm metadata `topic` chi tiết hơn `category`. (3) Dùng embedder thật (MiniLM) cho benchmark — mock embedder làm Q1/Q2 fail dù chunking hợp lý. (4) Viết gold answer kèm **tên section** để đánh giá top-1 chặt hơn top-3.

### Failure Analysis

| Query | Vấn đề | Nguyên nhân | Đề xuất |
|-------|--------|-------------|---------|
| Q1 | Top-3 không có customer_support_playbook | Mock embedder match từ "writing/chunking" nhầm sang chunking_report | Dùng embedder semantic; thêm `category=support` filter |
| Q2 | Không retrieve rag_system_design | Nhiều doc tiếng Anh kỹ thuật overlap từ khóa "system/retrieval" | Filter `doc_type=design`; tăng overlap hoặc index theo section header |

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | **81 / 100** |
