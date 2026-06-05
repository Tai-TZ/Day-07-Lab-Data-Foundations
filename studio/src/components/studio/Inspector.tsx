import { useState } from "react";
import type { Chunk, Document } from "@/lib/rag-types";
import type { NodeId, Phase } from "./types";

type Props = {
  selected: NodeId | null;
  phase: Phase;
  activeDoc: Document;
  allChunks: Chunk[];
  retrieved: Chunk[];
  query: string;
  queryEmbedding: number[];
  llmModel: string;
  embeddingModel: string;
  vectorDim: number;
  augmentedPrompt: string;
  sentences: { text: string; chunkId: string }[];
  generatedChars: number;
  highlightedChunkId: string | null;
  onHoverSentence: (id: string | null) => void;
};

export function Inspector(p: Props) {
  // Mode resolution
  let mode: "global" | "doc" | "chunks" | "chunk" | "vdb" | "query" | "qvec" | "topk" | "llm" | "answer";
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);

  switch (p.selected) {
    case "pdf":
    case "loader":
      mode = "doc";
      break;
    case "chunks":
    case "embedIdx":
      mode = "chunks";
      break;
    case "vdb":
      mode = "vdb";
      break;
    case "query":
    case "user":
      mode = "query";
      break;
    case "embedRet":
    case "qvec":
      mode = "qvec";
      break;
    case "topk":
      mode = "topk";
      break;
    case "llm":
      mode = "llm";
      break;
    case "answer":
      mode = "answer";
      break;
    default:
      // Auto-follow phase when nothing selected
      if (p.phase === "indexing") mode = "chunks";
      else if (p.phase === "embedding") mode = "qvec";
      else if (p.phase === "retrieving") mode = "topk";
      else if (p.phase === "generating" || p.phase === "done") mode = "answer";
      else mode = "global";
  }

  const selectedChunk = p.allChunks.find((c) => c.id === activeChunkId);

  return (
    <aside className="flex h-full flex-col overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <span className="mono-label">Inspector</span>
          <h3 className="mt-1 font-display text-base tracking-tight">{titleFor(mode)}</h3>
        </div>
        <span
          className="font-mono text-[10px] tracking-widest"
          style={{ color: accentFor(mode) }}
        >
          {modeLabel(mode)}
        </span>
      </header>

      <div className="flex-1 overflow-y-auto p-4">
        {mode === "global" && (
          <GlobalView
            embeddingModel={p.embeddingModel}
            llmModel={p.llmModel}
            allChunks={p.allChunks}
            vectorDim={p.vectorDim}
          />
        )}
        {mode === "doc" && <DocView doc={p.activeDoc} />}
        {mode === "chunks" && (
          <ChunksView
            doc={p.activeDoc}
            chunks={p.allChunks.filter((c) => c.docId === p.activeDoc.id)}
            activeChunkId={activeChunkId}
            onPick={setActiveChunkId}
            selectedChunk={selectedChunk}
          />
        )}
        {mode === "vdb" && (
          <VdbView
            allChunks={p.allChunks}
            queryEmbedding={p.queryEmbedding}
            retrieved={p.retrieved}
            vectorDim={p.vectorDim}
          />
        )}
        {mode === "query" && <QueryView query={p.query} />}
        {mode === "qvec" && <QvecView query={p.query} embedding={p.queryEmbedding} vectorDim={p.vectorDim} />}
        {mode === "topk" && (
          <TopKView
            retrieved={p.retrieved}
            highlightedChunkId={p.highlightedChunkId}
          />
        )}
        {mode === "llm" && (
          <LlmView
            llmModel={p.llmModel}
            prompt={
              p.augmentedPrompt ||
              (p.phase === "idle"
                ? "Run a query to see the augmented prompt sent to OpenRouter."
                : "Generating prompt…")
            }
          />
        )}
        {mode === "answer" && (
          <AnswerView
            sentences={p.sentences}
            generatedChars={p.generatedChars}
            highlightedChunkId={p.highlightedChunkId}
            onHoverSentence={p.onHoverSentence}
            phase={p.phase}
          />
        )}
      </div>
    </aside>
  );
}

function titleFor(mode: string) {
  return ({
    global: "Global settings",
    doc: "Document inspector",
    chunks: "Chunk inspector",
    vdb: "Vector database",
    query: "User query",
    qvec: "Query embedding",
    topk: "Top-k retrieved",
    llm: "Augmented prompt",
    answer: "Answer & grounding",
  } as Record<string, string>)[mode];
}
function modeLabel(mode: string) {
  return ({
    global: "SYSTEM",
    doc: "INDEXING",
    chunks: "INDEXING",
    vdb: "VECTOR DB",
    query: "RETRIEVAL",
    qvec: "RETRIEVAL",
    topk: "RETRIEVAL",
    llm: "GENERATION",
    answer: "GENERATION",
  } as Record<string, string>)[mode];
}
function accentFor(mode: string) {
  if (mode === "doc" || mode === "chunks") return "var(--indexing)";
  if (mode === "vdb") return "var(--mint)";
  if (mode === "global") return "var(--muted-foreground)";
  return "var(--retrieval)";
}

/* ---------------- Views ---------------- */

function GlobalView({
  embeddingModel,
  llmModel,
  allChunks,
  vectorDim,
}: {
  embeddingModel: string;
  llmModel: string;
  allChunks: Chunk[];
  vectorDim: number;
}) {
  const docs = new Set(allChunks.map((c) => c.docId)).size;
  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Chọn một khối trong sơ đồ để kiểm tra "ngóc ngách" tương ứng — dữ liệu thô, embedding, score, prompt và căn cứ.
      </p>
      <Stat label="Embedding model" value={embeddingModel} />
      <Stat label="LLM" value={llmModel} />
      <Stat label="Documents" value={String(docs)} />
      <Stat label="Total chunks" value={String(allChunks.length)} />
      <Stat label="Vector dim" value={String(vectorDim)} />
      <Stat label="Index" value="HNSW · cosine" />
    </div>
  );
}

function DocView({ doc }: { doc: Document }) {
  return (
    <div className="space-y-3">
      <Stat label="Filename" value={doc.name} />
      <Stat label="Type" value={`${doc.type} · ${doc.size}`} />
      <div>
        <div className="mono-label mb-2">Raw text</div>
        <div className="max-h-[60vh] overflow-y-auto rounded-md border border-border bg-background/40 p-3 font-mono text-[11px] leading-relaxed text-foreground/85">
          {doc.raw}
        </div>
      </div>
    </div>
  );
}

function ChunksView({
  doc,
  chunks,
  activeChunkId,
  onPick,
  selectedChunk,
}: {
  doc: Document;
  chunks: Chunk[];
  activeChunkId: string | null;
  onPick: (id: string) => void;
  selectedChunk?: Chunk;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="mono-label">{doc.name}</span>
        <span className="mono-label" style={{ color: "var(--indexing)" }}>
          {chunks.length} chunks
        </span>
      </div>
      <ul className="space-y-1.5">
        {chunks.map((c) => {
          const isActive = c.id === activeChunkId;
          return (
            <li key={c.id}>
              <button
                onClick={() => onPick(c.id)}
                className="block w-full rounded-md border p-2.5 text-left transition-all"
                style={{
                  borderColor: isActive ? "var(--indexing)" : "var(--border)",
                  backgroundColor: isActive
                    ? "color-mix(in oklab, var(--indexing) 8%, transparent)"
                    : "transparent",
                }}
              >
                <div className="flex items-center justify-between">
                  <span
                    className="mono-label"
                    style={{ color: isActive ? "var(--indexing)" : undefined }}
                  >
                    Chunk #{String(c.index + 1).padStart(3, "0")}
                  </span>
                  <span className="mono-label">{c.text.length} chars</span>
                </div>
                <p className="mt-1.5 line-clamp-2 text-[11.5px] leading-relaxed text-foreground/85">
                  {c.text}
                </p>
              </button>
            </li>
          );
        })}
      </ul>

      {selectedChunk && (
        <div className="mt-4 space-y-2 rounded-md border p-3"
          style={{ borderColor: "color-mix(in oklab, var(--indexing) 35%, var(--border))" }}>
          <div className="mono-label" style={{ color: "var(--indexing)" }}>Embedding vector</div>
          <VectorGrid vec={selectedChunk.embeddingPreview ?? []} />
          <SpaceMap chunks={chunks} highlighted={selectedChunk.id} />
        </div>
      )}
    </div>
  );
}

function VdbView({
  allChunks,
  queryEmbedding,
  retrieved,
  vectorDim,
}: {
  allChunks: Chunk[];
  queryEmbedding: number[];
  retrieved: Chunk[];
  vectorDim: number;
}) {
  return (
    <div className="space-y-3">
      <Stat label="Index type" value="HNSW · M=16 · efConstruction=200" />
      <Stat label="Distance" value="cosine" />
      <Stat label="Vectors" value={String(allChunks.length)} />
      <Stat label="Dimensions" value={String(vectorDim)} />
      <div>
        <div className="mono-label mb-2">Embedding space (2D projection)</div>
        <SpaceMap chunks={allChunks} retrieved={retrieved.map((r) => r.id)} queryEmbedding={queryEmbedding.length > 0} />
      </div>
    </div>
  );
}

function QueryView({ query }: { query: string }) {
  return (
    <div className="space-y-3">
      <div className="mono-label">Natural language</div>
      <p className="rounded-md border border-border bg-background/40 p-3 text-sm text-foreground">
        {query || "— enter a query in the left panel —"}
      </p>
      <Stat label="Tokens (est.)" value={String(Math.max(1, Math.round(query.length / 4)))} />
    </div>
  );
}

function QvecView({ query, embedding, vectorDim }: { query: string; embedding: number[]; vectorDim: number }) {
  if (!embedding.length) {
    return <p className="text-xs text-muted-foreground">Run a query to generate its embedding.</p>;
  }
  return (
    <div className="space-y-3">
      <Stat label="From query" value={`"${query.slice(0, 40)}${query.length > 40 ? "…" : ""}"`} />
      <Stat label="Dimensions" value={`${vectorDim} (showing ${Math.min(64, embedding.length)})`} />
      <VectorGrid vec={embedding.slice(0, 64)} />
    </div>
  );
}

function TopKView({ retrieved, highlightedChunkId }: { retrieved: Chunk[]; highlightedChunkId: string | null }) {
  if (!retrieved.length) {
    return <p className="text-xs text-muted-foreground">No retrieval yet. Send a query to see Top-k chunks.</p>;
  }
  return (
    <div className="space-y-2">
      {retrieved.map((c, i) => {
        const hi = highlightedChunkId === c.id;
        return (
          <div
            key={c.id}
            className="rounded-md border p-3 transition-all"
            style={{
              borderColor: hi ? "var(--retrieval)" : "var(--border)",
              backgroundColor: hi
                ? "color-mix(in oklab, var(--retrieval) 10%, transparent)"
                : "transparent",
            }}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span
                  className="rounded-sm px-1.5 py-0.5 font-mono text-[10px]"
                  style={{ backgroundColor: "color-mix(in oklab, var(--retrieval) 18%, transparent)", color: "var(--retrieval)" }}
                >
                  #{i + 1}
                </span>
                <span className="mono-label">{c.source} · p.{c.page}</span>
              </div>
              <ScoreBadge score={c.score} />
            </div>
            <p className="mt-2 text-[12px] leading-relaxed text-foreground/90">{c.text}</p>
          </div>
        );
      })}
    </div>
  );
}

function LlmView({ llmModel, prompt }: { llmModel: string; prompt: string }) {
  return (
    <div className="space-y-3">
      <Stat label="Model" value={llmModel} />
      <Stat label="Prompt length" value={`${prompt.length} chars`} />
      <div>
        <div className="mono-label mb-2">Augmented prompt sent to LLM</div>
        <pre className="max-h-[60vh] overflow-y-auto rounded-md border border-border bg-background/40 p-3 font-mono text-[10.5px] leading-relaxed text-foreground/85 whitespace-pre-wrap">
          {prompt}
        </pre>
      </div>
    </div>
  );
}

function AnswerView({
  sentences,
  generatedChars,
  highlightedChunkId,
  onHoverSentence,
  phase,
}: {
  sentences: { text: string; chunkId: string }[];
  generatedChars: number;
  highlightedChunkId: string | null;
  onHoverSentence: (id: string | null) => void;
  phase: Phase;
}) {
  if (!sentences.length) {
    return <p className="text-xs text-muted-foreground">Awaiting generation…</p>;
  }
  let consumed = 0;
  return (
    <div className="space-y-3">
      <div className="mono-label" style={{ color: "var(--retrieval)" }}>
        Hover một câu để xem chunk nguồn được highlight ở cột phải / dòng Top-k
      </div>
      <p className="rounded-md border border-border bg-background/40 p-3 text-sm leading-relaxed text-foreground">
        {sentences.map((s, i) => {
          const start = consumed;
          const end = consumed + s.text.length;
          const visible = Math.max(0, Math.min(s.text.length, generatedChars - start));
          consumed = end + 1;
          if (visible <= 0) return null;
          const isHi = highlightedChunkId === s.chunkId;
          return (
            <span
              key={i}
              onMouseEnter={() => visible === s.text.length && onHoverSentence(s.chunkId)}
              onMouseLeave={() => onHoverSentence(null)}
              className="cursor-pointer rounded-sm px-0.5 transition-colors"
              style={{
                backgroundColor: isHi ? "color-mix(in oklab, var(--retrieval) 22%, transparent)" : "transparent",
              }}
            >
              {s.text.slice(0, visible)}{" "}
            </span>
          );
        })}
        {phase === "generating" && (
          <span
            className="ml-0.5 inline-block h-3 w-1.5 -translate-y-px align-middle"
            style={{ backgroundColor: "var(--retrieval)", animation: "caret-blink 1s steps(1) infinite" }}
          />
        )}
      </p>
    </div>
  );
}

/* ---------------- Sub-views ---------------- */

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-background/40 px-3 py-2">
      <span className="mono-label">{label}</span>
      <span className="truncate text-right font-mono text-[11.5px] text-foreground">{value}</span>
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 1000) / 10;
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      <div className="h-1 w-14 overflow-hidden rounded-full bg-border">
        <div className="h-full" style={{ width: `${pct}%`, backgroundColor: "var(--retrieval)" }} />
      </div>
      <span className="font-mono text-[10.5px] tabular-nums" style={{ color: "var(--retrieval)" }}>
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

function VectorGrid({ vec }: { vec: number[] }) {
  return (
    <div className="grid grid-cols-8 gap-1 font-mono text-[9px]">
      {vec.map((v, i) => (
        <span
          key={i}
          className="rounded-sm px-1 py-1 text-center"
          style={{
            backgroundColor: `color-mix(in oklab, var(--mint) ${10 + Math.abs(v) * 50}%, transparent)`,
            color: "var(--foreground)",
          }}
        >
          {v.toFixed(2)}
        </span>
      ))}
    </div>
  );
}

function SpaceMap({
  chunks,
  highlighted,
  retrieved,
  queryEmbedding,
}: {
  chunks: Chunk[];
  highlighted?: string;
  retrieved?: string[];
  queryEmbedding?: boolean;
}) {
  // Map -1..1 to 0..100
  const toX = (v: number) => 50 + v * 42;
  const toY = (v: number) => 50 + v * 42;
  return (
    <div className="relative aspect-square w-full rounded-md border border-border bg-background/40">
      <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
        {/* axes */}
        <line x1="50" y1="0" x2="50" y2="100" stroke="var(--hairline)" strokeWidth="0.2" />
        <line x1="0" y1="50" x2="100" y2="50" stroke="var(--hairline)" strokeWidth="0.2" />
        {chunks.map((c) => {
          const isHi = c.id === highlighted;
          const isTop = retrieved?.includes(c.id);
          const r = isHi ? 1.5 : isTop ? 1.2 : 0.8;
          const fill = isHi
            ? "var(--indexing)"
            : isTop
              ? "var(--retrieval)"
              : "color-mix(in oklab, var(--mint) 50%, transparent)";
          return (
            <circle key={c.id} cx={toX(c.px)} cy={toY(c.py)} r={r} fill={fill} />
          );
        })}
        {queryEmbedding && (
          <g>
            <circle cx="50" cy="50" r="2" fill="var(--retrieval)" />
            <circle cx="50" cy="50" r="4" fill="none" stroke="var(--retrieval)" strokeWidth="0.3" />
          </g>
        )}
      </svg>
      <span className="mono-label absolute bottom-1 right-2">t-SNE · 2D</span>
    </div>
  );
}
