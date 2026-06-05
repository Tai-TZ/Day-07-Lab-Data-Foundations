import { useEffect, useState } from "react";
import { FileText, Link2, FileType, FileCode } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { DocumentUpload } from "@/components/rag/DocumentUpload";
import { fetchDocuments, type Document } from "@/lib/rag-api";

const ICONS: Record<Document["type"], React.ComponentType<{ className?: string; style?: React.CSSProperties }>> = {
  PDF: FileType,
  TXT: FileText,
  URL: Link2,
  MD: FileCode,
};

export function KnowledgeBase() {
  const { data: docs = [] } = useQuery({
    queryKey: ["rag-documents"],
    queryFn: fetchDocuments,
  });

  const [activeId, setActiveId] = useState<string | null>(null);
  const active = docs.find((d) => d.id === activeId) ?? docs[0];

  useEffect(() => {
    if (!activeId && docs[0]) setActiveId(docs[0].id);
  }, [docs, activeId]);

  return (
    <section id="knowledge" className="rounded-xl border border-border bg-card">
      <header className="flex items-baseline justify-between border-b border-border p-6">
        <div>
          <span className="mono-label">03 · Under the hood</span>
          <h2 className="mt-2 font-display text-3xl tracking-tight text-foreground">
            Knowledge base & chunking
          </h2>
        </div>
        <span className="mono-label hidden md:inline">
          {docs.length} docs · {docs.reduce((s, d) => s + d.chunks, 0)} chunks
        </span>
      </header>

      <DocumentUpload onUploaded={(docId) => setActiveId(docId)} />

      <div className="grid gap-0 md:grid-cols-[280px_1fr]">
        <ul className="border-b border-border md:border-b-0 md:border-r">
          {docs.map((d) => {
            const Icon = ICONS[d.type];
            const isActive = d.id === activeId;
            return (
              <li key={d.id}>
                <button
                  onClick={() => setActiveId(d.id)}
                  className="flex w-full items-start gap-3 border-b border-border px-5 py-4 text-left transition-colors hover:bg-background/40"
                  style={{
                    backgroundColor: isActive
                      ? "color-mix(in oklab, var(--mint) 6%, transparent)"
                      : undefined,
                  }}
                >
                  <Icon
                    className="mt-0.5 h-4 w-4 shrink-0"
                    style={{ color: isActive ? "var(--mint)" : "var(--muted-foreground)" }}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm text-foreground">
                      {d.name}
                      {d.uploaded && (
                        <span
                          className="mono-label ml-2 rounded-sm px-1.5 py-0.5"
                          style={{
                            backgroundColor: "color-mix(in oklab, var(--mint) 15%, transparent)",
                            color: "var(--mint)",
                          }}
                        >
                          uploaded
                        </span>
                      )}
                    </div>
                    <div className="mono-label mt-1">
                      {d.type} · {d.size} · {d.chunks} chunks
                    </div>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>

        <div className="p-6">
          <div className="mb-4 flex items-baseline justify-between">
            <div className="mono-label">Chunk preview</div>
            <div className="mono-label">size 512 · overlap 64</div>
          </div>
          <div className="space-y-2">
            {active?.preview?.map((text, i) => (
              <div
                key={i}
                className="group rounded-md border border-border bg-background/40 p-3 transition-colors hover:border-mint/40"
              >
                <div className="flex items-start gap-3">
                  <span
                    className="mono-label shrink-0 rounded-sm px-1.5 py-0.5"
                    style={{
                      backgroundColor: "color-mix(in oklab, var(--mint) 15%, transparent)",
                      color: "var(--mint)",
                    }}
                  >
                    #{String(i + 1).padStart(3, "0")}
                  </span>
                  <p className="text-sm leading-relaxed text-foreground/90">{text}</p>
                </div>
              </div>
            ))}
            {!active && (
              <p className="text-sm text-muted-foreground">No documents loaded from backend yet.</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
