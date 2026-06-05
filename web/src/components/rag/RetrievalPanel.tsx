import { Database, FileText } from "lucide-react";
import type { Chunk } from "@/lib/rag-api";

type Props = {
  chunks: Chunk[];
  active: boolean;
  done: boolean;
  highlightedChunkId?: string | null;
  onHoverChunk?: (id: string | null) => void;
};

export function RetrievalPanel({ chunks, active, done, highlightedChunkId, onHoverChunk }: Props) {
  return (
    <div className="rounded-xl border border-border bg-card p-6">
      <div className="mb-4 flex items-baseline justify-between">
        <h3 className="font-display text-xl tracking-tight text-foreground">Retrieval</h3>
        <span className="mono-label inline-flex items-center gap-1.5">
          <Database className="h-3 w-3" />
          Vector DB · Top-{chunks.length || 4}
        </span>
      </div>

      {!active && !done && chunks.length === 0 && (
        <EmptyState />
      )}

      {(active || done || chunks.length > 0) && (
        <div className="space-y-2">
          {(chunks.length ? chunks : Array.from({ length: 4 })).map((c, i) => {
            if (!c) return <SkeletonRow key={i} delay={i * 120} />;
            const chunk = c as Chunk;
            const highlighted = highlightedChunkId === chunk.id;
            return (
              <button
                key={chunk.id}
                onMouseEnter={() => onHoverChunk?.(chunk.id)}
                onMouseLeave={() => onHoverChunk?.(null)}
                className="group block w-full rounded-lg border bg-background/40 p-4 text-left transition-all"
                style={{
                  borderColor: highlighted
                    ? "var(--mint)"
                    : "var(--border)",
                  backgroundColor: highlighted
                    ? "color-mix(in oklab, var(--mint) 8%, var(--background))"
                    : undefined,
                }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-center gap-2 mono-label">
                    <FileText className="h-3 w-3" />
                    <span>{chunk.source}</span>
                    <span className="text-muted-foreground/60">· p.{chunk.page}</span>
                  </div>
                  <ScoreBadge score={chunk.score} />
                </div>
                <p className="mt-2 text-sm leading-relaxed text-foreground/90">{chunk.text}</p>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      <div className="h-1 w-16 overflow-hidden rounded-full bg-border">
        <div
          className="h-full"
          style={{
            width: `${pct}%`,
            backgroundColor: "var(--mint)",
          }}
        />
      </div>
      <span
        className="font-mono text-xs tabular-nums"
        style={{ color: "var(--mint)" }}
      >
        {pct}%
      </span>
    </div>
  );
}

function SkeletonRow({ delay }: { delay: number }) {
  return (
    <div
      className="rounded-lg border border-border bg-background/40 p-4"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-center justify-between">
        <div className="h-3 w-40 overflow-hidden rounded">
          <div className="shimmer h-full w-full" />
        </div>
        <div className="h-3 w-20 overflow-hidden rounded">
          <div className="shimmer h-full w-full" />
        </div>
      </div>
      <div className="mt-3 h-3 w-full overflow-hidden rounded">
        <div className="shimmer h-full w-full" />
      </div>
      <div className="mt-2 h-3 w-3/4 overflow-hidden rounded">
        <div className="shimmer h-full w-full" />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="rounded-lg border border-dashed border-border bg-background/20 p-8 text-center">
      <Database className="mx-auto h-6 w-6 text-muted-foreground/50" strokeWidth={1.5} />
      <p className="mt-3 text-sm text-muted-foreground">
        Submit a query to retrieve the most similar chunks from the vector store.
      </p>
    </div>
  );
}
