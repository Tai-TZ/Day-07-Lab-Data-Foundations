import { Upload, FileType, FileText, Link2, FileCode, Send, Settings2 } from "lucide-react";
import {
  SAMPLE_QUERIES,
  type ChunkConfig,
  type Document,
  type SplitterStrategy,
} from "@/lib/rag-types";
import { useRef, useState } from "react";

const ICONS: Record<Document["type"], React.ComponentType<{ className?: string; style?: React.CSSProperties }>> = {
  PDF: FileType,
  TXT: FileText,
  URL: Link2,
  MD: FileCode,
  DOCX: FileText,
};

type Props = {
  docs: Document[];
  activeDocId: string;
  onSelectDoc: (id: string) => void;
  chunkConfig: ChunkConfig;
  onChangeChunkConfig: (c: ChunkConfig) => void;
  embeddingModel: string;
  llmModels: string[];
  llmModel: string;
  onChangeLlm: (m: string) => void;
  topK: number;
  onChangeTopK: (k: number) => void;
  query: string;
  onChangeQuery: (q: string) => void;
  onRun: () => void;
  running: boolean;
  onUploadFile?: (file: File) => void | Promise<void>;
  uploading?: boolean;
  uploadMessage?: string | null;
  embeddingLocked?: boolean;
};

export function LeftPanel(p: Props) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (files: FileList | null) => {
    const file = files?.[0];
    if (!file || !p.onUploadFile) return;
    void p.onUploadFile(file);
  };

  return (
    <aside className="studio-scrollbar flex h-full min-h-0 flex-col gap-3 overflow-y-auto overscroll-contain">
      <SectionHeader index="01" label="Indexing config" color="var(--indexing)" />

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          handleFile(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className="cursor-pointer rounded-lg border border-dashed p-4 text-center transition-colors"
        style={{
          borderColor: drag ? "var(--indexing)" : "var(--border)",
          backgroundColor: drag ? "color-mix(in oklab, var(--indexing) 8%, transparent)" : "transparent",
        }}
      >
        <Upload className="mx-auto h-4 w-4" style={{ color: "var(--indexing)" }} />
        <p className="mt-2 text-xs text-foreground">
          {p.uploading ? "Đang tải lên…" : "Tải lên file .md hoặc .pdf"}
        </p>
        <p className="mono-label mt-1">click or drop here</p>
        {p.uploadMessage && (
          <p className="mono-label mt-2" style={{ color: "var(--mint)" }}>
            {p.uploadMessage}
          </p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".md,.pdf,text/markdown,application/pdf"
          className="hidden"
          onChange={(e) => {
            handleFile(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      <div className="rounded-lg border border-border bg-card/60">
        <div className="border-b border-border px-3 py-2">
          <span className="mono-label">Documents · {p.docs.length}</span>
        </div>
        <ul>
          {p.docs.map((d) => {
            const Icon = ICONS[d.type];
            const isActive = d.id === p.activeDocId;
            return (
              <li key={d.id}>
                <button
                  onClick={() => p.onSelectDoc(d.id)}
                  className="flex w-full items-center gap-2 border-b border-border/60 px-3 py-2 text-left transition-colors hover:bg-background/40"
                  style={{
                    backgroundColor: isActive
                      ? "color-mix(in oklab, var(--indexing) 8%, transparent)"
                      : undefined,
                  }}
                >
                  <Icon
                    className="h-3.5 w-3.5 shrink-0"
                    style={{ color: isActive ? "var(--indexing)" : "var(--muted-foreground)" }}
                  />
                  <span className="flex-1 truncate text-xs text-foreground">{d.name}</span>
                  <span className="mono-label">{d.type}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <Field label="Splitter strategy" icon={<Settings2 className="h-3 w-3" />}>
        <Select
          value={p.chunkConfig.strategy}
          onChange={(v) => p.onChangeChunkConfig({ ...p.chunkConfig, strategy: v as SplitterStrategy })}
          options={[
            { value: "recursive", label: "Recursive (default)" },
            { value: "character", label: "Character-based" },
            { value: "token", label: "Token-based" },
            { value: "sentence", label: "Sentence-based" },
          ]}
        />
      </Field>

      <Slider
        label="Chunk size"
        unit="tokens"
        min={64}
        max={1024}
        step={32}
        value={p.chunkConfig.size}
        onChange={(v) => p.onChangeChunkConfig({ ...p.chunkConfig, size: v })}
        accent="var(--indexing)"
      />
      <Slider
        label="Overlap"
        unit="tokens"
        min={0}
        max={200}
        step={8}
        value={p.chunkConfig.overlap}
        onChange={(v) => p.onChangeChunkConfig({ ...p.chunkConfig, overlap: v })}
        accent="var(--indexing)"
      />

      <Field label="Embedding model">
        <Select
          value={p.embeddingModel}
          onChange={() => {}}
          disabled={p.embeddingLocked}
          options={[{ value: p.embeddingModel, label: p.embeddingModel }]}
        />
      </Field>

      <div className="mt-2 h-px bg-border" />

      <SectionHeader index="02" label="Retrieval & generation" color="var(--retrieval)" />

      <div className="rounded-lg border border-border bg-card/60 p-3">
        <span className="mono-label">Query</span>
        <textarea
          value={p.query}
          onChange={(e) => p.onChangeQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              p.onRun();
            }
          }}
          rows={3}
          placeholder="Ask anything about your knowledge base…"
          className="mt-2 w-full resize-none rounded-md border border-border bg-background/60 px-3 py-2 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-mint/60"
        />
        <div className="mt-2 flex flex-wrap gap-1.5">
          {SAMPLE_QUERIES.map((s) => (
            <button
              key={s}
              onClick={() => p.onChangeQuery(s)}
              className="rounded-full border border-border bg-background/40 px-2 py-0.5 text-[10.5px] text-muted-foreground transition hover:border-retrieval/60 hover:text-foreground"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <Slider
        label="Top-k retrieval"
        unit="chunks"
        min={1}
        max={10}
        step={1}
        value={p.topK}
        onChange={p.onChangeTopK}
        accent="var(--retrieval)"
      />

      <Field label="LLM (OpenRouter)">
        <Select
          value={p.llmModel}
          onChange={p.onChangeLlm}
          disabled={p.llmModels.length === 0}
          options={
            p.llmModels.length > 0
              ? p.llmModels.map((m) => ({ value: m, label: m }))
              : [{ value: "", label: "Configure OPENROUTER_API_KEY in .env" }]
          }
        />
      </Field>

      <button
        onClick={p.onRun}
        disabled={p.running || !p.query.trim() || !p.llmModel}
        className="mt-1 inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40"
        style={{ backgroundColor: "var(--retrieval)", color: "white" }}
      >
        <Send className="h-3.5 w-3.5" />
        {p.running ? "Running pipeline…" : "Gửi truy vấn"}
      </button>
    </aside>
  );
}

function SectionHeader({ index, label, color }: { index: string; label: string; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="font-mono text-[10px] tracking-widest"
        style={{ color }}
      >
        {index}
      </span>
      <span className="mono-label" style={{ color }}>{label}</span>
      <div className="h-px flex-1" style={{ backgroundColor: `color-mix(in oklab, ${color} 35%, transparent)` }} />
    </div>
  );
}

function Field({ label, icon, children }: { label: string; icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div>
      <div className="mono-label mb-1.5 flex items-center gap-1.5">
        {icon}
        {label}
      </div>
      {children}
    </div>
  );
}

function Select({
  value,
  onChange,
  options,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-border bg-background/60 px-3 py-2 text-xs text-foreground outline-none focus:border-mint/60 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

function Slider({
  label,
  unit,
  min,
  max,
  step,
  value,
  onChange,
  accent,
}: {
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  accent: string;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="mono-label">{label}</span>
        <span className="font-mono text-xs tabular-nums" style={{ color: accent }}>
          {value} <span className="text-muted-foreground">{unit}</span>
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--accent-color)]"
        style={{ ["--accent-color" as string]: accent }}
      />
    </div>
  );
}
