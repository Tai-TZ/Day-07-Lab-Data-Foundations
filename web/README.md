# RAG Visualizer (Frontend)

Giao diện web trực quan hóa pipeline RAG, tích hợp với backend Python trong repo lab.

Nguồn UI gốc: [rag-reveal](https://github.com/NguyenTrongNguyen04/rag-reveal)

## Chạy

**Terminal 1 — Backend** (từ thư mục gốc lab):

```bash
pip install -r requirements.txt
python -m uvicorn rag_api:app --reload --port 8000
```

**Terminal 2 — Frontend** (từ thư mục `web/`):

```bash
npm install
cp .env.example .env
npm run dev
```

Mở trình duyệt theo URL Vite in ra (thường `http://localhost:5173`).

## Upload tài liệu

Trong mục **Knowledge base**, upload file `.md` của bạn (kéo-thả hoặc chọn file). File sẽ được lưu vào `data/uploads/` và index vào vector store để query ngay.

## Build production

```bash
npm run build
npm run preview
```
