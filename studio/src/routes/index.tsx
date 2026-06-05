import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { LeftPanel } from "@/components/studio/LeftPanel";
import { FlowCanvas } from "@/components/studio/FlowCanvas";
import { Inspector } from "@/components/studio/Inspector";
import type { NodeId, Phase } from "@/components/studio/types";
import { type Chunk, type ChunkConfig, type Document } from "@/lib/rag-types";
import {
  apiChunkToStudio,
  fetchIndexedChunks,
  fetchRagConfig,
  fetchStudioDocuments,
  reindexCatalog,
  runQuery,
  uploadDocument,
  waitForBackendReady,
} from "@/lib/rag-api";

export const Route = createFileRoute("/")({
  ssr: false,
  head: () => ({
    meta: [
      { title: "RAG Workflow Studio — Inspect every corner of the RAG pipeline" },
      {
        name: "description",
        content:
          "An interactive whiteboard for Retrieval-Augmented Generation. Configure chunking, embed, retrieve, augment, generate — and inspect raw data at every node.",
      },
      { property: "og:title", content: "RAG Workflow Studio" },
      {
        property: "og:description",
        content: "Open the black box of RAG. Click any node to inspect its raw data, embeddings, scores and prompts.",
      },
    ],
  }),
  component: Studio,
});

function Studio() {
  const [docs, setDocs] = useState<Document[]>([]);
  const [allChunks, setAllChunks] = useState<Chunk[]>([]);
  const [loadState, setLoadState] = useState<"loading" | "ready" | "error">("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [vectorDim, setVectorDim] = useState(384);
  const [llmBackend, setLlmBackend] = useState<"openrouter" | "unconfigured">("unconfigured");
  const [llmModels, setLlmModels] = useState<string[]>([]);

  const [activeDocId, setActiveDocId] = useState("");
  const [chunkConfig, setChunkConfig] = useState<ChunkConfig>({ strategy: "recursive", size: 512, overlap: 64 });
  const [embeddingModel, setEmbeddingModel] = useState("all-MiniLM-L6-v2");
  const [llmModel, setLlmModel] = useState("");
  const [topK, setTopK] = useState(4);
  const [query, setQuery] = useState("What is RAG?");
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [reindexing, setReindexing] = useState(false);
  const [loadingHint, setLoadingHint] = useState("Connecting to backend…");
  const [augmentedPrompt, setAugmentedPrompt] = useState("");

  const reloadChunks = useCallback(async () => {
    const chunks = await fetchIndexedChunks();
    setAllChunks(chunks);
  }, []);

  const reloadDocuments = async () => {
    setLoadState("loading");
    setLoadError(null);
    try {
      setLoadingHint("Waiting for MiniLM-L6-v2 model to load (first run may take 1–2 min)…");
      await waitForBackendReady();
      setLoadingHint("Loading documents…");
      const [config, next] = await Promise.all([fetchRagConfig(), fetchStudioDocuments()]);
      setEmbeddingModel(config.embedding_model);
      setVectorDim(config.vector_dim);
      setLlmBackend(config.llm);
      setLlmModels(config.llm_models ?? []);
      setLlmModel(config.llm_model || config.llm_models[0] || "");
      setChunkConfig({
        strategy: config.chunk_strategy,
        size: config.chunk_size,
        overlap: config.chunk_overlap,
      });
      setDocs(next);
      await reloadChunks();
      setLoadState("ready");
      if (next[0]) setActiveDocId((cur) => cur || next[0].id);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Failed to load documents");
      setLoadState("error");
    }
  };

  useEffect(() => {
    void reloadDocuments();
  }, []);

  const activeDoc: Document | undefined = docs.find((d) => d.id === activeDocId) ?? docs[0];

  const [phase, setPhase] = useState<Phase>("idle");
  const [queryEmbedding, setQueryEmbedding] = useState<number[]>([]);
  const [retrieved, setRetrieved] = useState<Chunk[]>([]);
  const [sentences, setSentences] = useState<{ text: string; chunkId: string }[]>([]);
  const [generatedChars, setGeneratedChars] = useState(0);
  const [selected, setSelected] = useState<NodeId | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const timers = useRef<number[]>([]);
  const raf = useRef<number>(0);
  const reindexTimer = useRef<number>(0);
  const skipReindex = useRef(true);

  useEffect(() => () => {
    timers.current.forEach(clearTimeout);
    cancelAnimationFrame(raf.current);
    clearTimeout(reindexTimer.current);
  }, []);

  useEffect(() => {
    if (loadState !== "ready") return;
    if (skipReindex.current) {
      skipReindex.current = false;
      return;
    }
    setPhase("indexing");
    setReindexing(true);
    clearTimeout(reindexTimer.current);
    reindexTimer.current = window.setTimeout(() => {
      void (async () => {
        try {
          await reindexCatalog(chunkConfig);
          await reloadChunks();
        } catch (err) {
          setUploadMessage(err instanceof Error ? err.message : "Reindex failed");
        } finally {
          setReindexing(false);
          setPhase((p) => (p === "indexing" ? "idle" : p));
        }
      })();
    }, 600);
    return () => clearTimeout(reindexTimer.current);
  }, [chunkConfig.strategy, chunkConfig.size, chunkConfig.overlap, loadState, reloadChunks]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setUploadMessage(null);
    try {
      const res = await uploadDocument(file);
      setUploadMessage(res.message);
      await reloadDocuments();
      setActiveDocId(res.document.id);
    } catch (err) {
      setUploadMessage(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const runPipeline = () => {
    if (!query.trim() || docs.length === 0 || !llmModel) return;
    timers.current.forEach(clearTimeout);
    cancelAnimationFrame(raf.current);
    timers.current = [];

    setSelected(null);
    setRetrieved([]);
    setSentences([]);
    setGeneratedChars(0);
    setQueryEmbedding([]);
    setAugmentedPrompt("");

    setPhase("embedding");

    timers.current.push(
      window.setTimeout(async () => {
        setPhase("retrieving");
        try {
          const res = await runQuery(query, topK, llmModel);
          const scored = res.chunks.map(apiChunkToStudio);
          setQueryEmbedding(res.query_embedding ?? []);
          setRetrieved(scored);
          setAugmentedPrompt(res.prompt ?? "");
          setPhase("generating");

          const answerSentences = res.sentences;
          setSentences(answerSentences);
          const total = answerSentences.map((s) => s.text).join(" ").length;
          const start = performance.now();
          const dur = 2400;
          const tick = (t: number) => {
            const p = Math.min(1, (t - start) / dur);
            setGeneratedChars(Math.floor(total * p));
            if (p < 1) raf.current = requestAnimationFrame(tick);
            else setPhase("done");
          };
          raf.current = requestAnimationFrame(tick);
        } catch (err) {
          setUploadMessage(err instanceof Error ? err.message : "Query failed");
          setPhase("idle");
        }
      }, 400),
    );
  };

  const running =
    phase === "embedding" || phase === "retrieving" || phase === "generating" || reindexing;

  if (loadState === "loading") {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 bg-background text-muted-foreground">
        <p>Loading knowledge base…</p>
        <p className="mono-label">{loadingHint}</p>
        <p className="mono-label">API: {import.meta.env.VITE_RAG_API_URL ?? "/api (proxy)"}</p>
      </div>
    );
  }

  if (loadState === "error") {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-background px-6 text-center">
        <p className="text-foreground">Cannot load knowledge base</p>
        <p className="max-w-md text-sm text-muted-foreground">{loadError}</p>
        <p className="text-sm text-muted-foreground">
          Start backend: <code className="text-foreground">python -m uvicorn rag_api:app --reload --port 8000</code>
        </p>
        <button
          type="button"
          onClick={() => void reloadDocuments()}
          className="rounded-lg px-4 py-2 text-sm"
          style={{ backgroundColor: "var(--retrieval)", color: "white" }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (!activeDoc) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 bg-background px-6 text-center">
        <p className="text-foreground">No documents in the knowledge base yet.</p>
        <p className="text-sm text-muted-foreground">
          Upload a <code>.md</code> or <code>.pdf</code> file to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="h-screen overflow-hidden bg-background text-foreground">
      <TopBar embeddingModel={embeddingModel} llmModel={llmModel} llmBackend={llmBackend} />
      <div className="grid h-[calc(100vh-52px)] grid-cols-[280px_minmax(0,1fr)_360px] gap-3 p-3">
        <div className="min-h-0 overflow-hidden rounded-xl border border-border bg-card/70 p-3">
          <LeftPanel
            docs={docs}
            activeDocId={activeDocId}
            onSelectDoc={setActiveDocId}
            chunkConfig={chunkConfig}
            onChangeChunkConfig={setChunkConfig}
            embeddingModel={embeddingModel}
            llmModels={llmModels}
            llmModel={llmModel}
            onChangeLlm={setLlmModel}
            topK={topK}
            onChangeTopK={setTopK}
            query={query}
            onChangeQuery={setQuery}
            onRun={runPipeline}
            running={running}
            onUploadFile={handleUpload}
            uploading={uploading}
            uploadMessage={uploadMessage}
            embeddingLocked
          />
        </div>

        <FlowCanvas
          phase={phase}
          selected={selected}
          onSelect={(n) => setSelected((cur) => (cur === n ? null : n))}
          chunkCount={allChunks.filter((c) => c.docId === activeDocId).length}
          topK={topK}
        />

        <Inspector
          selected={selected}
          phase={phase}
          activeDoc={activeDoc}
          allChunks={allChunks}
          retrieved={retrieved}
          query={query}
          queryEmbedding={queryEmbedding}
          llmModel={llmModel}
          embeddingModel={embeddingModel}
          vectorDim={vectorDim}
          augmentedPrompt={augmentedPrompt}
          sentences={sentences}
          generatedChars={generatedChars}
          highlightedChunkId={hovered}
          onHoverSentence={setHovered}
        />
      </div>
    </div>
  );
}

function TopBar({
  embeddingModel,
  llmModel,
  llmBackend,
}: {
  embeddingModel: string;
  llmModel: string;
  llmBackend: string;
}) {
  return (
    <header className="flex h-[52px] items-center justify-between border-b border-border px-4">
      <div className="flex items-center gap-3">
        <div
          className="flex h-7 w-7 items-center justify-center rounded-md"
          style={{
            background: "linear-gradient(135deg, var(--indexing), var(--retrieval))",
          }}
        >
          <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4 text-background">
            <path d="M4 7h16M4 12h10M4 17h16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
          </svg>
        </div>
        <div className="leading-none">
          <div className="font-display text-sm tracking-tight">RAG Workflow Studio</div>
          <div className="mono-label mt-0.5">
            {embeddingModel} · {llmBackend === "openrouter" ? llmModel : "LLM not configured"}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <Legend dot="var(--indexing)" label="Indexing" />
        <Legend dot="var(--retrieval)" label="Retrieval" />
        <Legend dot="var(--mint)" label="Vector DB" />
      </div>
    </header>
  );
}

function Legend({ dot, label }: { dot: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: dot }} />
      <span className="mono-label">{label}</span>
    </div>
  );
}
