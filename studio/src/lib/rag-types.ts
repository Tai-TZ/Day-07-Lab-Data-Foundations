export type Chunk = {
  id: string;
  docId: string;
  index: number;
  source: string;
  page: number;
  text: string;
  score: number;
  px: number;
  py: number;
  embeddingPreview?: number[];
};

export type Document = {
  id: string;
  name: string;
  type: "PDF" | "TXT" | "URL" | "MD" | "DOCX";
  size: string;
  raw: string;
};

export type SplitterStrategy = "character" | "token" | "sentence" | "recursive";

export type ChunkConfig = {
  strategy: SplitterStrategy;
  size: number;
  overlap: number;
};

export const SAMPLE_QUERIES = [
  "What is RAG?",
  "Why does chunk size matter?",
  "How does similarity search work?",
];
