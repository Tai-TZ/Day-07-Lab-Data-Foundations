export type Chunk = {
  id: string;
  source: string;
  page: number;
  text: string;
  score: number;
  used?: boolean;
};

export type Document = {
  id: string;
  name: string;
  type: "PDF" | "TXT" | "URL" | "MD";
  size: string;
  chunks: number;
  preview: string[];
  uploaded?: boolean;
};

export type UploadResponse = {
  document: Document;
  message: string;
};

export type QueryResponse = {
  chunks: Chunk[];
  answer: string;
  sentences: { text: string; chunkId: string }[];
};

function apiBase() {
  return (import.meta.env.VITE_RAG_API_URL as string | undefined) ?? "http://localhost:8000";
}

export async function fetchDocuments(): Promise<Document[]> {
  const res = await fetch(`${apiBase()}/documents`);
  if (!res.ok) throw new Error(`Failed to fetch documents: ${res.status}`);
  return (await res.json()) as Document[];
}

export async function runQuery(query: string, top_k = 4): Promise<QueryResponse> {
  const res = await fetch(`${apiBase()}/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, top_k }),
  });
  if (!res.ok) throw new Error(`Failed to query: ${res.status}`);
  return (await res.json()) as QueryResponse;
}

export async function uploadMarkdown(file: File): Promise<UploadResponse> {
  if (!file.name.toLowerCase().endsWith(".md")) {
    throw new Error("Only .md files are supported.");
  }

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${apiBase()}/documents/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = `Upload failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // ignore parse errors
    }
    throw new Error(detail);
  }

  return (await res.json()) as UploadResponse;
}

