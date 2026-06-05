# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thành Tài  
**MSSV:** 2A202600627  
**Nhóm:** A3  

**Thành viên nhóm:**
- Nguyễn Trọng Nguyên — 2A202600548
- Nguyễn Thành Tài — 2A202600627
- Ngô Thị Ánh — 2A202600979

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

**Domain:** Research Paper Knowledge Base — bộ preprint khoa học đa lĩnh vực (thống kê, kinh tế lượng, dịch tễ, tính toán, ML, RL)

**Tại sao nhóm chọn domain này?**
> Nhóm sử dụng 6 preprint từ thư mục `Research Paper/`, chuyển PDF → `.txt` và lưu vào `data/`. Corpus phản ánh use case thực tế: trợ lý tra cứu paper với abstract, methodology và kết quả benchmark. Domain đa lĩnh vực cho phép kiểm tra metadata filter theo `category` (vd. `machine_learning` cho câu hỏi về GARD/LoRA).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | paper_I_random_matrix.txt | Research Paper (PDF→txt) | 54,622 | category=statistics, topic=random_matrix, language=en, doc_type=preprint |
| 2 | paper_II_weak_iv_estimators.txt | Research Paper (PDF→txt) | 42,390 | category=econometrics, topic=instrumental_variables, language=en, doc_type=preprint |
| 3 | paper_III_sir_seir_identifiability.txt | Research Paper (PDF→txt) | 56,000 | category=epidemiology, topic=compartmental_models, language=en, doc_type=preprint |
| 4 | paper_IV_krylov_preconditioners.txt | Research Paper (PDF→txt) | 51,431 | category=numerical_methods, topic=sparse_linear_systems, language=en, doc_type=preprint |
| 5 | paper_V_gard_lora.txt | Research Paper (PDF→txt) | 55,603 | category=machine_learning, topic=parameter_efficient_finetuning, language=en, doc_type=preprint |
| 6 | paper_VI_lace_exploration.txt | Research Paper (PDF→txt) | 31,079 | category=reinforcement_learning, topic=exploration, language=en, doc_type=preprint |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | statistics, econometrics, machine_learning | Lọc câu hỏi theo lĩnh vực, tránh retrieve nhầm paper khác chủ đề |
| `topic` | string | random_matrix, instrumental_variables | Phân biệt chi tiết hơn trong cùng category |
| `language` | string | en | Chuẩn bị mở rộng đa ngôn ngữ |
| `doc_type` | string | preprint | Phân biệt loại tài liệu (preprint, report, notes) |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu (`chunk_size=200`):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| paper_I_random_matrix.txt | FixedSizeChunker (`fixed_size`) | 304 | 199.6 | Trung bình — cắt giữa đoạn/ công thức |
| paper_I_random_matrix.txt | SentenceChunker (`by_sentences`) | 140 | 389.1 | Tốt hơn — giữ câu, nhưng chunk dài |
| paper_I_random_matrix.txt | RecursiveChunker (`recursive`) | 358 | 151.3 | Tốt — nhiều chunk nhỏ, coherent hơn |
| paper_III_sir_seir_identifiability.txt | FixedSizeChunker (`fixed_size`) | 311 | 200.0 | Trung bình |
| paper_III_sir_seir_identifiability.txt | SentenceChunker (`by_sentences`) | 162 | 344.6 | Tốt |
| paper_III_sir_seir_identifiability.txt | RecursiveChunker (`recursive`) | 374 | 148.5 | Tốt |
| paper_V_gard_lora.txt | FixedSizeChunker (`fixed_size`) | 309 | 199.9 | Trung bình |
| paper_V_gard_lora.txt | SentenceChunker (`by_sentences`) | 135 | 410.7 | Tốt |
| paper_V_gard_lora.txt | RecursiveChunker (`recursive`) | 366 | 150.7 | Tốt |

### Strategy Của Tôi

**Loại:** FixedSizeChunker (`chunk_size=300`, `overlap=50`)

**Mô tả cách hoạt động:**
> Sliding window: mỗi chunk tối đa 300 ký tự, overlap 50 ký tự, bước nhảy 250. Phù hợp paper dài (~30K–56K ký tự) vì đơn giản, dễ scale, không phụ thuộc cấu trúc section của từng preprint.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Research papers có cấu trúc tương đối đồng nhất (Abstract → Introduction → Methods) nhưng text extract từ PDF có line-break artifacts. Fixed-size xử lý đồng nhất mọi paper, tạo **1167 chunks** cho 6 papers — ít hơn recursive (1290) trong khi vẫn đạt 5/5 benchmark.

**Code snippet:**
```python
chunker = FixedSizeChunker(chunk_size=300, overlap=50)
chunks = chunker.chunk(paper_text)
```

### So Sánh: Strategy của tôi vs Baseline

Benchmark retrieval trên toàn bộ corpus (6 papers) với embedder `all-MiniLM-L6-v2`:

| Corpus | Strategy | Chunk Count | Retrieval Quality? |
|--------|----------|-------------|--------------------|
| 6 research papers | **fixed_size_300 (của tôi)** | 1167 | **5/5 relevant top-3** |
| 6 research papers | sentence_2 | 1199 | 5/5 relevant top-3 |
| 6 research papers | recursive_300 | 1290 | 5/5 relevant top-3 |

> Cả 3 strategy đều 5/5 trên corpus paper. Fixed_size tạo ít chunk nhất, phù hợp prototype; recursive có top-1 tốt hơn trên một số query (vd. GARD Q4: 0.790 vs 0.759).

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| **Tôi** | FixedSizeChunker (300) | 10/10 | Ít chunk, ổn định trên paper dài | Cắt giữa câu; artifact PDF |
| Nguyễn Trọng Nguyên | RecursiveChunker (300) | 10/10 | Chunk coherent hơn trên section | Nhiều chunk hơn (~10%) |
| Ngô Thị Ánh | SentenceChunker (2 câu) | 10/10 | Giữ câu học thuật trọn vẹn | Chunk dài, tốn embed |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với MiniLM, cả 3 đều 10/10. **RecursiveChunker** có lợi thế khi paper nhiều section/heading. **FixedSizeChunker** vẫn hợp lý cho corpus lớn cần index nhanh và đơn giản.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của tôi khi implement các phần chính trong package `src`.

### Chunking Functions

**`FixedSizeChunker.chunk`** — approach:
> `range(0, len(text), chunk_size - overlap)` với overlap giữ ngữ cảnh ở ranh giới — quan trọng vì PDF→txt hay cắt giữa từ/công thức.

**`SentenceChunker.chunk`** — approach:
> Regex tách câu sau `.!?`, gom `max_sentences_per_chunk` câu. Hữu ích cho abstract/introduction nhưng chunk dài trên paper ~50K ký tự.

**`RecursiveChunker.chunk`** — approach:
> Đệ quy separator `["\n\n", "\n", ". ", " ", ""]`, fallback FixedSize khi hết separator.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Embed và lưu in-memory; search bằng cosine similarity. ChromaDB chỉ bật khi set `CHROMA_PERSIST_DIR`.

**`search_with_filter` + `delete_document`** — approach:
> Filter metadata trước (vd. `category=machine_learning` cho Q4 GARD), rồi mới tính similarity.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top-k → build prompt với citation → gọi `llm_fn`. Với paper dài, top-k=3 thường đủ nếu retrieval đúng paper.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
collected 42 items — 42 passed in 0.05s
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

**Kết quả nào bất ngờ nhất?**
> Pair 5 có score âm — embedding biểu diễn hướng ngữ nghĩa, không khớp từ khóa. Cặp cùng chủ đề đều > 0.6 dù không trùng từ.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries: **FixedSizeChunker(300, overlap=50)** + **all-MiniLM-L6-v2**.  
Lệnh: `py phase2_benchmark.py --embedder local --my-strategy fixed_size_300 --export report/phase2_results.json`

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is MPCX and what finite-dimensional correction problem does it study? | MPCX compares finite-dimensional corrections to the Marchenko-Pastur law for Gaussian Wishart matrices (dims 50–5000). |
| 2 | Which IV estimators formed the lowest-loss cluster in the IVX benchmark? | LIML, Fuller(1), and Fuller(4) formed a low-loss cluster (~0.84); 2SLS and JIVE were much worse. |
| 3 | What is PRIM and what identifiability diagnostics does it compare for epidemic models? | PRIM benchmarks SIR/SEIR identifiability via profile likelihood, Fisher Information, and MLE on synthetic outbreaks. |
| 4 | How does GARD allocate LoRA ranks using gradient spectral analysis? | GARD uses per-layer gradient covariance spectra to allocate discrete LoRA ranks under a fixed adapter budget. (filter: `category=machine_learning`) |
| 5 | Which Krylov preconditioners were compared on sparse linear systems? | Jacobi, SSOR, ILU, and AMG with CG, GMRES, and BiCGSTAB. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Nguồn |
|---|-------|--------------------------------|-------|-----------|-------|
| 1 | MPCX finite-dimensional correction | "...how finite-dimensional spectra deviate from MP..." | 0.736 | ✅ | paper_I_random_matrix |
| 2 | IV estimators lowest-loss cluster | "Figure 1: Overview of the IVX evaluation framework..." | 0.635 | ✅ | paper_II_weak_iv_estimators |
| 3 | PRIM identifiability diagnostics | "...practical identifiability in compartmental epidemic models..." | 0.804 | ✅ | paper_III_sir_seir_identifiability |
| 4 | GARD LoRA rank allocation | "...introduced GARD, a gradient-aware rank allocation framework..." | 0.759 | ✅ | paper_V_gard_lora |
| 5 | Krylov preconditioners compared | "Comparative Analysis of Krylov Preconditioners on Sparse Linear Systems..." | 0.904 | ✅ | paper_IV_krylov_preconditioners |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **5 / 5**

---

## 7. What I Learned (5 điểm — Demo)

### Failure Analysis (Ex 3.5)

**Query chưa tối ưu grounding:** Q2 — *"Which IV estimators formed the lowest-loss cluster in the IVX benchmark?"*

**Hiện tượng:**
> Top-1 đúng paper (`paper_II_weak_iv_estimators`) nhưng chunk là **Figure 1 caption** ("Overview of the IVX evaluation framework"), không phải đoạn abstract nêu rõ LIML/Fuller/2SLS. Score 0.635 — thấp hơn Q5 (0.904).

**Nguyên nhân:**
> (1) PDF→txt tạo artifact line-break (`cent 2` thay vì "Figure 2"). (2) Fixed-size 300 cắt giữa abstract và figure references. (3) Query hỏi kết quả cụ thể nhưng top-1 là mô tả pipeline.

**Đề xuất cải thiện:**
> (1) Preprocess PDF text: gộp line-break giữa câu. (2) Tăng `chunk_size` lên 512 cho paper dài. (3) Thêm metadata `section=abstract|results` khi parse. (4) Dùng `category=econometrics` filter cho câu IV.

### Bài Học Từ Nhóm

**Điều hay nhất tôi học được từ thành viên khác:**
> Recursive chunking giữ section tốt hơn trên paper markdown-style; sentence chunking giữ abstract trọn vẹn.

**Điều hay nhất từ nhóm khác (demo):**
> Metadata filter `category=machine_learning` giúp Q4 GARD không bị nhiễu bởi paper ML khác (LACE).

**Nếu làm lại:**
> (1) Clean PDF text trước khi index. (2) Hybrid chunking: recursive cho `.txt` paper, fixed cho notes ngắn. (3) Thêm `topic` filter ngoài `category`.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **86 / 100** |
