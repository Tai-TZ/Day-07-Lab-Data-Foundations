import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { QueryInput } from "@/components/rag/QueryInput";
import { Stepper, type Step } from "@/components/rag/Stepper";
import { EmbeddingPanel } from "@/components/rag/EmbeddingPanel";
import { RetrievalPanel } from "@/components/rag/RetrievalPanel";
import { GenerationPanel } from "@/components/rag/GenerationPanel";
import { KnowledgeBase } from "@/components/rag/KnowledgeBase";
import { History, type HistoryItem } from "@/components/rag/History";
import { runQuery, type Chunk } from "@/lib/rag-api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "RAG Visualizer — See how Retrieval-Augmented Generation thinks" },
      {
        name: "description",
        content:
          "An interactive, transparent UI for Retrieval-Augmented Generation. Watch embeddings, vector retrieval and LLM generation happen in real time.",
      },
      { property: "og:title", content: "RAG Visualizer" },
      {
        property: "og:description",
        content: "Open the black box of RAG. Visualize embedding, retrieval and generation step by step.",
      },
    ],
  }),
  component: Index,
});

type Phase = "idle" | "embedding" | "retrieving" | "generating" | "done";

function Index() {
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [sentences, setSentences] = useState<{ text: string; chunkId: string }[]>([]);
  const [hovered, setHovered] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const timers = useRef<number[]>([]);
  const pipelineRef = useRef<HTMLDivElement>(null);
  const pending = useRef<Promise<{ chunks: Chunk[]; sentences: { text: string; chunkId: string }[]; answer: string }> | null>(
    null,
  );

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const run = (q: string) => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setQuery(q);
    setChunks([]);
    setSentences([]);
    setHovered(null);
    setPhase("embedding");

    pipelineRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });

    pending.current = runQuery(q, 4);

    timers.current.push(
      window.setTimeout(() => {
        setPhase("retrieving");
        timers.current.push(
          window.setTimeout(async () => {
            try {
              const res = await pending.current;
              if (!res) return;
              setChunks(res.chunks);
            } catch {
              setChunks([]);
            }
          }, 350),
        );
      }, 1200),
    );

    timers.current.push(
      window.setTimeout(() => {
        (async () => {
          try {
            const res = await pending.current;
            if (!res) return;
            setSentences(res.sentences);
          } catch {
            setSentences([]);
          } finally {
            setPhase("generating");
          }
        })();
      }, 2900),
    );

    timers.current.push(
      window.setTimeout(() => {
        setPhase("done");
        (async () => {
          try {
            const res = await pending.current;
            if (!res) return;
            const answer = res.answer || res.sentences.map((s) => s.text).join(" ");
            setHistory((h) => [{ id: crypto.randomUUID(), query: q, answer, ts: Date.now() }, ...h].slice(0, 8));
          } catch {
            setHistory((h) => [{ id: crypto.randomUUID(), query: q, answer: "(error)", ts: Date.now() }, ...h].slice(0, 8));
          }
        })();
      }, 5400),
    );
  };

  const steps: Step[] = useMemo(() => {
    const status = (s: Phase): Step["status"] => {
      const order: Phase[] = ["embedding", "retrieving", "generating", "done"];
      const cur = order.indexOf(phase);
      const tgt = order.indexOf(s);
      if (phase === "idle") return "idle";
      if (cur > tgt) return "done";
      if (cur === tgt) return s === "done" ? "done" : "active";
      return "idle";
    };
    return [
      { id: "embed", label: "Embedding query", sublabel: "encode", status: status("embedding") },
      { id: "retrieve", label: "Searching vector DB", sublabel: "top-k", status: status("retrieving") },
      { id: "generate", label: "Generating answer", sublabel: "stream", status: status("generating") },
    ];
  }, [phase]);

  const isRunning = phase !== "idle" && phase !== "done";

  return (
    <div className="min-h-screen bg-background">
      <TopBar />

      <main className="mx-auto max-w-[1400px] px-6 pb-24 pt-10 md:px-10">
        <Hero />

        <div className="mt-12 grid gap-6 lg:grid-cols-[1fr_300px]">
          <section className="space-y-6">
            <div className="rounded-2xl border border-border bg-card p-6">
              <div className="mb-4 flex items-baseline justify-between">
                <span className="mono-label">01 · Ask</span>
                <span className="mono-label">enter to send · shift+enter for newline</span>
              </div>
              <QueryInput onSubmit={run} disabled={isRunning} />
            </div>

            <div ref={pipelineRef} className="scroll-mt-8 space-y-6">
              <div className="flex items-baseline justify-between">
                <div>
                  <span className="mono-label">02 · The pipeline</span>
                  <h2 className="mt-2 font-display text-3xl tracking-tight">
                    RAG, in motion.
                  </h2>
                </div>
              </div>

              <Stepper steps={steps} />

              <EmbeddingPanel
                query={query || "—"}
                active={phase === "embedding"}
                done={phase === "retrieving" || phase === "generating" || phase === "done"}
              />

              <RetrievalPanel
                chunks={chunks}
                active={phase === "retrieving"}
                done={phase === "generating" || phase === "done"}
                highlightedChunkId={hovered}
                onHoverChunk={setHovered}
              />

              <GenerationPanel
                query={query}
                chunks={chunks}
                sentences={sentences}
                active={phase === "generating"}
                done={phase === "done"}
                onHoverSentence={setHovered}
                highlightedChunkId={hovered}
              />
            </div>
          </section>

          <div className="lg:sticky lg:top-6 lg:self-start">
            <History items={history} />
          </div>
        </div>

        <div className="mt-12">
          <KnowledgeBase />
        </div>

        <Footer />
      </main>
    </div>
  );
}

function TopBar() {
  return (
    <header className="border-b border-border">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 py-4 md:px-10">
        <div className="flex items-center gap-3">
          <div
            className="flex h-8 w-8 items-center justify-center rounded-md"
            style={{ backgroundColor: "var(--mint)" }}
          >
            <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4" style={{ color: "var(--background)" }}>
              <path d="M4 7h16M4 12h10M4 17h16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
            </svg>
          </div>
          <div className="leading-none">
            <div className="font-display text-sm tracking-tight">rag.visualizer</div>
            <div className="mono-label mt-1">v0.1 · explainable AI</div>
          </div>
        </div>
        <nav className="hidden items-center gap-6 md:flex">
          <a className="text-sm text-muted-foreground hover:text-foreground" href="#pipeline">Pipeline</a>
          <a className="text-sm text-muted-foreground hover:text-foreground" href="#knowledge">Knowledge</a>
          <a
            className="rounded-full px-4 py-1.5 text-sm"
            href="https://arxiv.org/abs/2005.11401"
            target="_blank"
            rel="noreferrer"
            style={{ backgroundColor: "var(--foreground)", color: "var(--background)" }}
          >
            Read the paper
          </a>
        </nav>
      </div>
    </header>
  );
}

function Hero() {
  return (
    <section className="relative overflow-hidden rounded-2xl border border-border bg-card">
      <div className="hairline-grid absolute inset-0 opacity-30" />
      <div className="relative grid gap-8 p-10 md:grid-cols-[1.4fr_1fr] md:p-14">
        <div>
          <span className="mono-label inline-flex items-center gap-2">
            <span
              className="pulse-dot inline-block h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: "var(--mint)" }}
            />
            live · explainable retrieval
          </span>
          <h1 className="mt-6 font-display text-5xl leading-[1.02] tracking-tight md:text-[64px]">
            Open the black box{" "}
            <span style={{ color: "var(--mint)" }}>of RAG.</span>
          </h1>
          <p className="mt-5 max-w-xl text-base leading-relaxed text-muted-foreground md:text-lg">
            Ask a question and watch the system embed your query, scan a vector
            store for the most relevant chunks, augment a prompt and stream a
            grounded answer — every step rendered in real time.
          </p>
          <div className="mt-7 flex flex-wrap items-center gap-3">
            <a
              href="#pipeline"
              className="rounded-full px-5 py-2.5 text-sm font-medium"
              style={{ backgroundColor: "var(--foreground)", color: "var(--background)" }}
            >
              Try a query →
            </a>
            <a
              href="#knowledge"
              className="rounded-full border border-border px-5 py-2.5 text-sm text-foreground hover:border-mint/40"
            >
              Inspect the knowledge base
            </a>
          </div>
        </div>

        <div className="relative">
          <div className="absolute inset-0 -z-10 rounded-xl"
            style={{
              background: "radial-gradient(80% 60% at 70% 30%, color-mix(in oklab, var(--mint) 18%, transparent), transparent 70%)",
            }}
          />
          <div className="rounded-xl border border-border bg-background/60 p-5 font-mono text-[11px] leading-relaxed">
            <div className="mono-label mb-3">trace · q-7f3a</div>
            {[
              ["embed", "query → vec[1536]", "12ms"],
              ["search", "HNSW · cosine · k=4", "38ms"],
              ["augment", "prompt += context[4]", "2ms"],
              ["generate", "stream tokens", "1.8s"],
            ].map(([k, v, t]) => (
              <div key={k} className="grid grid-cols-[80px_1fr_auto] items-center gap-3 border-t border-border py-2 first:border-t-0">
                <span style={{ color: "var(--mint)" }}>{k}</span>
                <span className="text-foreground/80">{v}</span>
                <span className="text-muted-foreground">{t}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="mt-20 border-t border-border pt-8">
      <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-center">
        <p className="mono-label">
          rag.visualizer · built to make retrieval transparent
        </p>
        <p className="text-xs text-muted-foreground">
          Powered by Day-07 lab backend · mock embeddings by default.
        </p>
      </div>
    </footer>
  );
}
