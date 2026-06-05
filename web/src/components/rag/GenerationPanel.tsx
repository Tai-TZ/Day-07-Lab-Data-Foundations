import { useEffect, useState } from "react";
import { Sparkles } from "lucide-react";
import type { Chunk } from "@/lib/rag-api";

type Sentence = { text: string; chunkId: string };

type Props = {
  query: string;
  chunks: Chunk[];
  sentences: Sentence[];
  active: boolean;
  done: boolean;
  onHoverSentence?: (chunkId: string | null) => void;
  highlightedChunkId?: string | null;
};

export function GenerationPanel({
  query,
  chunks,
  sentences,
  active,
  done,
  onHoverSentence,
  highlightedChunkId,
}: Props) {
  return (
    <div className="rounded-xl border border-border bg-card p-6">
      <div className="mb-4 flex items-baseline justify-between">
        <h3 className="font-display text-xl tracking-tight text-foreground">
          Augmentation & Generation
        </h3>
        <span className="mono-label inline-flex items-center gap-1.5">
          <Sparkles className="h-3 w-3" />
          LLM · streaming
        </span>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <PromptPreview query={query} chunks={chunks} />
        <StreamedAnswer
          sentences={sentences}
          active={active}
          done={done}
          onHoverSentence={onHoverSentence}
          highlightedChunkId={highlightedChunkId}
        />
      </div>
    </div>
  );
}

function PromptPreview({ query, chunks }: { query: string; chunks: Chunk[] }) {
  return (
    <div className="rounded-md border border-border bg-background/40 p-4 font-mono text-[12px] leading-relaxed">
      <div className="mono-label mb-2">Augmented prompt</div>
      <div className="space-y-2 text-foreground/85">
        <p className="text-muted-foreground"># System</p>
        <p>Answer using only the context below. Cite sources.</p>
        <p className="text-muted-foreground"># Context</p>
        {chunks.length ? (
          chunks.map((c, i) => (
            <p key={c.id} className="truncate">
              <span style={{ color: "var(--mint)" }}>[{i + 1}]</span>{" "}
              {c.text.slice(0, 90)}…
            </p>
          ))
        ) : (
          <p className="text-muted-foreground">[awaiting retrieval]</p>
        )}
        <p className="text-muted-foreground"># Question</p>
        <p>{query || "—"}</p>
      </div>
    </div>
  );
}

function StreamedAnswer({
  sentences,
  active,
  done,
  onHoverSentence,
  highlightedChunkId,
}: {
  sentences: Sentence[];
  active: boolean;
  done: boolean;
  onHoverSentence?: (chunkId: string | null) => void;
  highlightedChunkId?: string | null;
}) {
  const [chars, setChars] = useState(0);
  const full = sentences.map((s) => s.text).join(" ");

  useEffect(() => {
    if (!active && !done) {
      setChars(0);
      return;
    }
    if (done) {
      setChars(full.length);
      return;
    }
    setChars(0);
    let raf = 0;
    const start = performance.now();
    const speed = full.length / 2200; // ~2.2s
    const tick = (t: number) => {
      const c = Math.min(full.length, Math.floor((t - start) * speed));
      setChars(c);
      if (c < full.length) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, done, full]);

  // Reveal sentences progressively based on chars
  let consumed = 0;
  const revealed: { sentence: Sentence; visibleText: string; fullyShown: boolean }[] = [];
  for (const s of sentences) {
    const startIdx = consumed;
    const endIdx = consumed + s.text.length;
    if (chars <= startIdx) {
      revealed.push({ sentence: s, visibleText: "", fullyShown: false });
    } else if (chars >= endIdx) {
      revealed.push({ sentence: s, visibleText: s.text, fullyShown: true });
    } else {
      revealed.push({ sentence: s, visibleText: s.text.slice(0, chars - startIdx), fullyShown: false });
    }
    consumed = endIdx + 1;
  }

  return (
    <div className="rounded-md border border-border bg-background/40 p-4">
      <div className="mono-label mb-2">Generated answer</div>
      {!active && !done && (
        <p className="text-sm text-muted-foreground">
          The model will read the context and stream a grounded response here.
        </p>
      )}
      {(active || done) && (
        <p className="text-sm leading-relaxed">
          {revealed.map(({ sentence, visibleText, fullyShown }, i) => {
            if (!visibleText) return null;
            const isHi = highlightedChunkId === sentence.chunkId;
            return (
              <span
                key={i}
                onMouseEnter={() => fullyShown && onHoverSentence?.(sentence.chunkId)}
                onMouseLeave={() => onHoverSentence?.(null)}
                className="cursor-pointer rounded-sm px-0.5 transition-colors"
                style={{
                  backgroundColor: isHi
                    ? "color-mix(in oklab, var(--mint) 22%, transparent)"
                    : "transparent",
                  color: "var(--foreground)",
                }}
              >
                {visibleText}{" "}
              </span>
            );
          })}
          {!done && (
            <span
              className="ml-0.5 inline-block h-3 w-1.5 -translate-y-px align-middle"
              style={{
                backgroundColor: "var(--mint)",
                animation: "caret-blink 1s steps(1) infinite",
              }}
            />
          )}
        </p>
      )}
      {done && (
        <p className="mono-label mt-4">
          Hover a sentence to see its source chunk →
        </p>
      )}
    </div>
  );
}
