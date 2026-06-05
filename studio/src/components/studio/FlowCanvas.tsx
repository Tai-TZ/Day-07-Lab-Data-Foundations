import { Database, FileText, Sparkles, User, Boxes, ScanSearch, MessageCircleQuestion, Cpu, Layers } from "lucide-react";
import type { Phase, NodeId } from "./types";

type Props = {
  phase: Phase;
  selected: NodeId | null;
  onSelect: (n: NodeId) => void;
  chunkCount: number;
  topK: number;
};

// Node positions in % of viewbox (1000x560)
const POS: Record<NodeId, { x: number; y: number; w: number; h: number; label: string; icon: any; accent: "indexing" | "retrieval" | "mint" }> = {
  pdf:        { x: 40,  y: 60,  w: 90,  h: 70, label: "PDF",            icon: FileText,  accent: "indexing" },
  loader:     { x: 180, y: 60,  w: 130, h: 70, label: "Doc Loader",     icon: Layers,    accent: "indexing" },
  chunks:     { x: 360, y: 60,  w: 130, h: 70, label: "Text Chunks",    icon: Boxes,     accent: "indexing" },
  embedIdx:   { x: 540, y: 60,  w: 140, h: 70, label: "Embedding Model",icon: Cpu,       accent: "indexing" },
  vdb:        { x: 800, y: 200, w: 140, h: 180, label: "Vector DB",     icon: Database,  accent: "mint" },
  user:       { x: 40,  y: 410, w: 90,  h: 70, label: "User",           icon: User,      accent: "retrieval" },
  query:      { x: 180, y: 410, w: 110, h: 70, label: "Query",          icon: MessageCircleQuestion, accent: "retrieval" },
  embedRet:   { x: 320, y: 410, w: 140, h: 70, label: "Embedding Model",icon: Cpu,       accent: "retrieval" },
  qvec:       { x: 500, y: 410, w: 130, h: 70, label: "Query Embedding",icon: ScanSearch,accent: "retrieval" },
  topk:       { x: 500, y: 500, w: 130, h: 50, label: "Top-k chunks",   icon: Boxes,     accent: "retrieval" },
  llm:        { x: 320, y: 500, w: 110, h: 50, label: "LLM",            icon: Sparkles,  accent: "retrieval" },
  answer:     { x: 180, y: 500, w: 110, h: 50, label: "Answer",         icon: Sparkles,  accent: "retrieval" },
};

// Edges
type Edge = { from: NodeId; to: NodeId; label?: string; activeOn: Phase[] };
const EDGES: Edge[] = [
  { from: "pdf", to: "loader", activeOn: ["indexing"] },
  { from: "loader", to: "chunks", activeOn: ["indexing"] },
  { from: "chunks", to: "embedIdx", activeOn: ["indexing"] },
  { from: "embedIdx", to: "vdb", label: "embeddings", activeOn: ["indexing"] },
  { from: "user", to: "query", activeOn: ["embedding", "retrieving", "generating", "done"] },
  { from: "query", to: "embedRet", activeOn: ["embedding", "retrieving", "generating", "done"] },
  { from: "embedRet", to: "qvec", activeOn: ["embedding", "retrieving", "generating", "done"] },
  { from: "qvec", to: "vdb", label: "similarity", activeOn: ["retrieving", "generating", "done"] },
  { from: "vdb", to: "topk", label: "top-k", activeOn: ["retrieving", "generating", "done"] },
  { from: "topk", to: "llm", activeOn: ["generating", "done"] },
  { from: "llm", to: "answer", activeOn: ["generating", "done"] },
  { from: "answer", to: "user", activeOn: ["done"] },
];

const ACCENT: Record<"indexing" | "retrieval" | "mint", string> = {
  indexing: "var(--indexing)",
  retrieval: "var(--retrieval)",
  mint: "var(--mint)",
};

export function FlowCanvas({ phase, selected, onSelect, chunkCount, topK }: Props) {
  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border border-border bg-card">
      <div className="hairline-grid absolute inset-0 opacity-25" />

      {/* Section labels */}
      <div className="absolute left-4 top-3 z-10 flex items-center gap-2">
        <span className="font-mono text-[10px] tracking-widest" style={{ color: "var(--indexing)" }}>
          ▲ INDEXING
        </span>
        <span className="mono-label">offline · build the vector store</span>
      </div>
      <div className="absolute bottom-3 left-4 z-10 flex items-center gap-2">
        <span className="font-mono text-[10px] tracking-widest" style={{ color: "var(--retrieval)" }}>
          ▼ RETRIEVAL & GENERATION
        </span>
        <span className="mono-label">online · answer the user</span>
      </div>

      <svg viewBox="0 0 1000 560" className="absolute inset-0 h-full w-full" preserveAspectRatio="xMidYMid meet">
        {/* Divider */}
        <line x1="20" y1="285" x2="980" y2="285"
          stroke="color-mix(in oklab, var(--coral) 60%, transparent)"
          strokeWidth="1" strokeDasharray="6 8" />

        {/* Edges */}
        {EDGES.map((e, i) => {
          const a = POS[e.from];
          const b = POS[e.to];
          const ax = a.x + a.w / 2;
          const ay = a.y + a.h / 2;
          const bx = b.x + b.w / 2;
          const by = b.y + b.h / 2;
          const active = e.activeOn.includes(phase);
          const accentColor = active ? ACCENT[a.accent] : "color-mix(in oklab, var(--hairline) 70%, transparent)";
          return (
            <g key={i}>
              <line
                x1={ax} y1={ay} x2={bx} y2={by}
                stroke={accentColor}
                strokeWidth={active ? 1.5 : 1}
                className={active ? "dash-flow" : ""}
                opacity={active ? 0.9 : 0.5}
              />
              {e.label && (
                <text
                  x={(ax + bx) / 2}
                  y={(ay + by) / 2 - 6}
                  textAnchor="middle"
                  className="font-mono"
                  fontSize="9"
                  fill="var(--muted-foreground)"
                  style={{ letterSpacing: "0.12em", textTransform: "uppercase" }}
                >
                  {e.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Nodes — absolutely positioned in % */}
      <div className="absolute inset-0">
        {(Object.keys(POS) as NodeId[]).map((id) => {
          const n = POS[id];
          const isSelected = selected === id;
          const isActiveNode = nodeActive(id, phase);
          return (
            <button
              key={id}
              onClick={() => onSelect(id)}
              className="absolute flex flex-col items-center justify-center gap-1 rounded-md border text-center transition-all"
              style={{
                left: `${(n.x / 1000) * 100}%`,
                top: `${(n.y / 560) * 100}%`,
                width: `${(n.w / 1000) * 100}%`,
                height: `${(n.h / 560) * 100}%`,
                backgroundColor: isSelected
                  ? `color-mix(in oklab, ${ACCENT[n.accent]} 18%, var(--card))`
                  : "color-mix(in oklab, var(--card) 80%, var(--background))",
                borderColor: isSelected || isActiveNode
                  ? ACCENT[n.accent]
                  : "var(--border)",
                color: "var(--foreground)",
                ...(isActiveNode ? { ["--glow" as string]: ACCENT[n.accent] } : {}),
              }}
              data-active={isActiveNode}
            >
              <span
                className={isActiveNode ? "node-pulse" : ""}
                style={{ color: ACCENT[n.accent] }}
              >
                <n.icon className="h-4 w-4" />
              </span>
              <span className="px-1 text-[11px] font-medium leading-tight">{n.label}</span>
              {id === "chunks" && (
                <span className="font-mono text-[9px]" style={{ color: ACCENT[n.accent] }}>
                  ×{chunkCount}
                </span>
              )}
              {id === "topk" && (
                <span className="font-mono text-[9px]" style={{ color: ACCENT[n.accent] }}>
                  k={topK}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Phase indicator */}
      <div className="absolute right-4 top-3 z-10 flex items-center gap-2 rounded-full border border-border bg-background/70 px-3 py-1 backdrop-blur">
        <span
          className="inline-block h-1.5 w-1.5 rounded-full pulse-dot"
          style={{ backgroundColor: phase === "idle" ? "var(--muted-foreground)" : "var(--mint)" }}
        />
        <span className="mono-label">phase · {phase}</span>
      </div>
    </div>
  );
}

function nodeActive(id: NodeId, phase: Phase): boolean {
  if (phase === "indexing") return ["pdf", "loader", "chunks", "embedIdx", "vdb"].includes(id);
  if (phase === "embedding") return ["user", "query", "embedRet", "qvec"].includes(id);
  if (phase === "retrieving") return ["qvec", "vdb", "topk"].includes(id);
  if (phase === "generating") return ["topk", "llm"].includes(id);
  if (phase === "done") return ["answer", "user"].includes(id);
  return false;
}
