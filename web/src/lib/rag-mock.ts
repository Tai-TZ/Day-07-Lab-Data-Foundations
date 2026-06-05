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
};

export const DOCUMENTS: Document[] = [
  {
    id: "doc-1",
    name: "rag-architecture-whitepaper.pdf",
    type: "PDF",
    size: "1.2 MB",
    chunks: 42,
    preview: [
      "Retrieval-Augmented Generation (RAG) combines a parametric language model with a non-parametric memory store.",
      "The retriever encodes the user query into a dense vector and performs an approximate nearest neighbor search over the corpus.",
      "Top-K passages are concatenated to the prompt before being fed to the generator, grounding output in source material.",
      "Chunk size, overlap, and embedding model are the three knobs that most affect retrieval quality.",
      "Hybrid retrieval blends dense vectors with BM25 keyword signals to improve recall on rare terms.",
    ],
  },
  {
    id: "doc-2",
    name: "vector-databases-2025.md",
    type: "MD",
    size: "82 KB",
    chunks: 18,
    preview: [
      "Vector databases store high-dimensional embeddings and support similarity search via HNSW or IVF indexes.",
      "Cosine similarity is the dominant distance metric for text embeddings produced by transformer encoders.",
      "Popular choices in 2025 include pgvector, Pinecone, Weaviate, Qdrant, and Milvus.",
      "Filtering on metadata at query time can be done pre-filter or post-filter — each has latency tradeoffs.",
    ],
  },
  {
    id: "doc-3",
    name: "chunking-strategies.txt",
    type: "TXT",
    size: "24 KB",
    chunks: 12,
    preview: [
      "Fixed-size chunking splits text by token count with optional overlap of 10-20%.",
      "Semantic chunking uses sentence embeddings to detect topic boundaries before splitting.",
      "For technical docs, structural chunking by heading hierarchy preserves the most context.",
    ],
  },
  {
    id: "doc-4",
    name: "https://arxiv.org/abs/2005.11401",
    type: "URL",
    size: "—",
    chunks: 28,
    preview: [
      "The original RAG paper from Facebook AI introduces a sequence-to-sequence model with a learned retriever.",
      "RAG-Token and RAG-Sequence differ in whether retrieval happens once or per generation step.",
    ],
  },
];

const CHUNK_POOL: Omit<Chunk, "score" | "id">[] = [
  { source: "rag-architecture-whitepaper.pdf", page: 3, text: "Retrieval-Augmented Generation grounds an LLM in external knowledge by injecting retrieved passages into the prompt before generation. This reduces hallucinations and lets the model cite specific sources." },
  { source: "rag-architecture-whitepaper.pdf", page: 7, text: "The retriever encodes the user query into a dense vector using the same embedding model as the corpus, then performs approximate nearest neighbor search to surface the Top-K most similar chunks." },
  { source: "vector-databases-2025.md", page: 1, text: "Modern vector databases use HNSW (Hierarchical Navigable Small World) graphs to achieve sub-millisecond similarity search across millions of embeddings with 95%+ recall." },
  { source: "chunking-strategies.txt", page: 1, text: "Chunk size directly affects retrieval precision. Smaller chunks (256 tokens) improve specificity; larger chunks (1024 tokens) preserve more surrounding context for the generator." },
  { source: "rag-architecture-whitepaper.pdf", page: 12, text: "Augmentation concatenates the retrieved chunks with the original question into a single prompt template, often prefixed with an instruction like 'Answer using only the context below.'" },
  { source: "vector-databases-2025.md", page: 4, text: "Cosine similarity is preferred over Euclidean distance for text embeddings because direction in the embedding space encodes semantic meaning more robustly than magnitude." },
  { source: "chunking-strategies.txt", page: 2, text: "Overlap of 10-20% between adjacent chunks prevents key sentences from being split across boundaries and lost to the retriever." },
];

export function fakeEmbedding(text: string, dims = 24): number[] {
  // deterministic pseudo-embedding for nice visuals
  const vec: number[] = [];
  let seed = 0;
  for (let i = 0; i < text.length; i++) seed = (seed * 31 + text.charCodeAt(i)) >>> 0;
  for (let i = 0; i < dims; i++) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    vec.push(((seed % 2000) / 1000 - 1));
  }
  return vec;
}

export function retrieveChunks(query: string, k = 4): Chunk[] {
  const q = query.toLowerCase();
  const scored = CHUNK_POOL.map((c, i) => {
    const words = q.split(/\s+/).filter(Boolean);
    let hits = 0;
    for (const w of words) if (c.text.toLowerCase().includes(w)) hits++;
    const base = 0.62 + Math.random() * 0.05;
    const boost = Math.min(0.35, hits * 0.08);
    return {
      id: `chunk-${i}`,
      source: c.source,
      page: c.page,
      text: c.text,
      score: Math.min(0.98, base + boost),
    };
  });
  return scored.sort((a, b) => b.score - a.score).slice(0, k);
}

export function composeAnswer(query: string, chunks: Chunk[]): { sentences: { text: string; chunkId: string }[] } {
  // Map answer sentences to source chunks
  const sentences = [
    { text: `RAG (Retrieval-Augmented Generation) is a pattern that grounds an LLM in external knowledge by retrieving relevant context at query time.`, chunkId: chunks[0]?.id ?? "chunk-0" },
    { text: `When you ask a question, the system first encodes your query into a dense vector and searches a vector database for the Top-K most similar passages.`, chunkId: chunks[1]?.id ?? chunks[0]?.id ?? "chunk-1" },
    { text: `Those passages are then concatenated with your original question into an augmented prompt, which the language model uses to generate a grounded answer.`, chunkId: chunks[2]?.id ?? chunks[0]?.id ?? "chunk-2" },
    { text: `The quality of the answer depends heavily on chunk size, embedding model, and the similarity metric — typically cosine similarity over HNSW indexes.`, chunkId: chunks[3]?.id ?? chunks[0]?.id ?? "chunk-3" },
  ];
  void query;
  return { sentences };
}

export const SAMPLE_QUERIES = [
  "How does Retrieval-Augmented Generation actually work?",
  "What is the role of a vector database in RAG?",
  "Why does chunk size matter for retrieval quality?",
];
