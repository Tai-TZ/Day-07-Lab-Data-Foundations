## RAG Visualizer — Hướng dẫn chạy

Giao diện web nằm trong thư mục `web/`, kết nối với backend `rag_api.py` dùng code lab (`src/`) và dữ liệu trong `data/`.

### 1) Chạy backend (Python)

Từ thư mục gốc `Day-07-Lab-Data-Foundations`:

```bash
pip install -r requirements.txt
python -m uvicorn rag_api:app --reload --port 8000
```

Kiểm tra nhanh:

- `GET http://localhost:8000/health`
- `GET http://localhost:8000/documents`

### 2) Chạy frontend (UI)

```bash
cd web
npm install
cp .env.example .env
npm run dev
```

Mở UI theo log Vite (thường `http://localhost:5173`).

### 3) Upload tài liệu `.md`

Trên UI (mục **Knowledge base**), kéo-thả hoặc chọn file `.md` để thêm vào vector store.

- File được lưu tại `data/uploads/`
- Tự động chunk + embed ngay sau upload
- Giới hạn: UTF-8, tối đa 5 MB

### 4) Dữ liệu hiển thị

Backend load tất cả file `.md`/`.txt` trong `data/` và `data/uploads/`, chunk theo:

- `chunk_size=512`
- `overlap=64`

Embedding mặc định: `_mock_embed` (không cần OpenAI/local model).

Xem thêm: `web/README.md`
