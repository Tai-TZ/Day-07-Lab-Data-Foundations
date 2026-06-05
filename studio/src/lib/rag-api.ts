import type { Chunk, ChunkConfig, Document } from "@/lib/rag-types";

export type ApiDocument = Document & {
  chunks?: number;
  preview?: string[];
  uploaded?: boolean;
};

export type ApiChunk = {
  id: string;
  doc_id?: string;
  chunk_index?: number;
  source: string;
  page: number;
  text: string;
  score: number;
  px?: number;
  py?: number;
  embedding_preview?: number[];
};

export type QueryResponse = {
  chunks: ApiChunk[];
  answer: string;
  sentences: { text: string; chunkId: string }[];
  query_embedding?: number[];
  prompt?: string;
  llm_model?: string;
};

export type UploadResponse = {
  document: ApiDocument;
  message: string;
};

export type RagConfig = {
  embedding_model: string;
  vector_dim: number;
  chunk_strategy: ChunkConfig["strategy"];
  chunk_size: number;
  chunk_overlap: number;
  llm: "openrouter" | "unconfigured";
  llm_model: string;
  llm_models: string[];
  documents: number;
  chunks: number;
};

function apiBase() {
  const fromEnv = import.meta.env.VITE_RAG_API_URL as string | undefined;
  if (fromEnv) return fromEnv.replace(/\/$/, "");
  // Dev proxy in vite.config.ts avoids browser CORS issues
  if (import.meta.env.DEV) return "/api";
  return "http://localhost:8000";
}

export function apiChunkToStudio(chunk: ApiChunk): Chunk {
  return {
    id: chunk.id,
    docId: chunk.doc_id || "",
    index: chunk.chunk_index ?? 0,
    source: chunk.source,
    page: chunk.page,
    text: chunk.text,
    score: chunk.score,
    px: chunk.px ?? 0,
    py: chunk.py ?? 0,
    embeddingPreview: chunk.embedding_preview,
  };
}

export type HealthResponse = {
  ok: boolean;
  status?: "ready" | "loading";
  error?: string;
  embedding_model?: string;
  vector_dim?: number;
};

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${apiBase()}/health`);
  if (!res.ok) throw new Error(`Failed to fetch health (${res.status})`);
  return (await res.json()) as HealthResponse;
}

export async function waitForBackendReady(maxAttempts = 90, delayMs = 2000): Promise<HealthResponse> {
  let lastError = "Backend not ready";
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      const health = await fetchHealth();
      if (health.ok) return health;
      lastError = health.error || "Loading embedding model (all-MiniLM-L6-v2)…";
    } catch (err) {
      lastError = err instanceof Error ? err.message : "Cannot reach API";
    }
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  throw new Error(lastError);
}

export async function fetchRagConfig(): Promise<RagConfig> {
  const base = apiBase();
  let res: Response;
  try {
    res = await fetch(`${base}/config`);
  } catch {
    throw new Error(`Cannot reach API at ${base}. Start backend: cd Day-07-Lab-Data-Foundations && python -m uvicorn rag_api:app --reload --port 8000`);
  }
  if (!res.ok) {
    if (res.status === 503) {
      await waitForBackendReady();
      return fetchRagConfig();
    }
    throw new Error(`Failed to fetch config (${res.status}) from ${base}`);
  }
  return (await res.json()) as RagConfig;
}

export async function fetchStudioDocuments(): Promise<Document[]> {
  const base = apiBase();
  let res: Response;
  try {
    res = await fetch(`${base}/documents`);
  } catch {
    throw new Error(`Cannot reach API at ${base}. Is uvicorn running?`);
  }
  if (!res.ok) throw new Error(`Failed to fetch documents (${res.status}) from ${base}`);
  const rows = (await res.json()) as ApiDocument[];
  return rows.map((d) => ({
    id: d.id,
    name: d.name,
    type: (d.type === "PDF" || d.type === "TXT" || d.type === "URL" || d.type === "MD" || d.type === "DOCX"
      ? d.type
      : "MD") as Document["type"],
    size: d.size,
    raw: d.raw || "",
  }));
}

export async function fetchIndexedChunks(docId?: string): Promise<Chunk[]> {
  const url = docId ? `${apiBase()}/chunks?doc_id=${encodeURIComponent(docId)}` : `${apiBase()}/chunks`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch chunks (${res.status})`);
  const rows = (await res.json()) as ApiChunk[];
  return rows.map(apiChunkToStudio);
}

export async function reindexCatalog(config: ChunkConfig): Promise<void> {
  const res = await fetch(`${apiBase()}/reindex`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      strategy: config.strategy,
      size: config.size,
      overlap: config.overlap,
    }),
  });
  if (!res.ok) throw new Error(`Reindex failed (${res.status})`);
}

export async function runQuery(query: string, top_k = 4, llm_model?: string): Promise<QueryResponse> {
  const res = await fetch(`${apiBase()}/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, top_k, llm_model }),
  });
  if (!res.ok) {
    let detail = `Query failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return (await res.json()) as QueryResponse;
}

export async function embedText(text: string): Promise<number[]> {
  const res = await fetch(`${apiBase()}/embed`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`Failed to embed text: ${res.status}`);
  const body = (await res.json()) as { embedding: number[] };
  return body.embedding;
}

const UPLOAD_EXTENSIONS = [".md", ".pdf"] as const;

function isUploadableFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return UPLOAD_EXTENSIONS.some((ext) => name.endsWith(ext));
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  if (!isUploadableFile(file)) {
    throw new Error("Only .md and .pdf files are supported.");
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
      // ignore
    }
    throw new Error(detail);
  }

  return (await res.json()) as UploadResponse;
}

/** @deprecated Use uploadDocument */
export async function uploadMarkdown(file: File): Promise<UploadResponse> {
  return uploadDocument(file);
}
