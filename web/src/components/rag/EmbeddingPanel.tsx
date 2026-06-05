import { useEffect, useState } from "react";
import { fakeEmbedding } from "@/lib/rag-mock";

type Props = { query: string; active: boolean; done: boolean };

export function EmbeddingPanel({ query, active, done }: Props) {
  const [progress, setProgress] = useState(0);
  const vec = fakeEmbedding(query, 32);

  useEffect(() => {
    if (!active && !done) {
      setProgress(0);
      return;
    }
    if (done) {
      setProgress(1);
      return;
    }
    setProgress(0);
    const start = performance.now();
    const dur = 1100;
    let raf = 0;
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / dur);
      setProgress(p);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, done, query]);

  const revealCount = Math.floor(vec.length * progress);

  return (
    <Section title="Embedding" caption="Câu hỏi → Dense vector">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_auto_1fr] md:items-center">
        <div className="rounded-md border border-border bg-background/40 p-4">
          <div className="mono-label mb-2">Natural language</div>
          <p className="text-sm leading-relaxed text-foreground">"{query}"</p>
        </div>

        <Arrow active={active} />

        <div className="rounded-md border border-border bg-background/40 p-4">
          <div className="mono-label mb-2 flex items-center justify-between">
            <span>Vector · dim 1536</span>
            <span style={{ color: "var(--mint)" }}>
              {Math.round(progress * 100)}%
            </span>
          </div>
          <div className="grid grid-cols-8 gap-1 font-mono text-[10px] leading-none">
            {vec.map((v, i) => {
              const shown = i < revealCount || done;
              return (
                <span
                  key={i}
                  className="rounded-sm px-1 py-1 text-center transition-colors"
                  style={{
                    backgroundColor: shown
                      ? `color-mix(in oklab, var(--mint) ${10 + Math.abs(v) * 40}%, transparent)`
                      : "color-mix(in oklab, var(--hairline) 40%, transparent)",
                    color: shown ? "var(--foreground)" : "transparent",
                  }}
                >
                  {v.toFixed(2)}
                </span>
              );
            })}
          </div>
        </div>
      </div>
    </Section>
  );
}

function Section({ title, caption, children }: { title: string; caption: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-card p-6">
      <div className="mb-4 flex items-baseline justify-between">
        <h3 className="font-display text-xl tracking-tight text-foreground">{title}</h3>
        <span className="mono-label">{caption}</span>
      </div>
      {children}
    </div>
  );
}

function Arrow({ active }: { active: boolean }) {
  return (
    <div className="flex items-center justify-center md:px-2">
      <div
        className="h-px w-full md:w-16"
        style={{
          background: active
            ? "linear-gradient(90deg, transparent, var(--mint), transparent)"
            : "var(--hairline)",
        }}
      />
    </div>
  );
}
