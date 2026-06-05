import { MessageSquare } from "lucide-react";

export type HistoryItem = { id: string; query: string; answer: string; ts: number };

export function History({ items, onSelect }: { items: HistoryItem[]; onSelect?: (id: string) => void }) {
  return (
    <aside className="rounded-xl border border-border bg-card p-5">
      <div className="mb-4 flex items-center justify-between">
        <span className="mono-label inline-flex items-center gap-1.5">
          <MessageSquare className="h-3 w-3" /> History
        </span>
        <span className="mono-label">{items.length}</span>
      </div>
      {items.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Past conversations will appear here.
        </p>
      )}
      <ul className="space-y-3">
        {items.map((it) => (
          <li key={it.id}>
            <button
              onClick={() => onSelect?.(it.id)}
              className="block w-full rounded-md border border-border bg-background/40 p-3 text-left transition hover:border-mint/40"
            >
              <p className="line-clamp-2 text-xs text-foreground">{it.query}</p>
              <p className="mono-label mt-2 truncate">
                {new Date(it.ts).toLocaleTimeString()} · {it.answer.length} chars
              </p>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}
