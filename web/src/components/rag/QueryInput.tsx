import { ArrowUp, Sparkles } from "lucide-react";
import { useState } from "react";

const SAMPLE_QUERIES = [
  "Summarize the key information from the loaded files.",
  "What is a vector database used for in RAG?",
  "Why does chunk size and overlap matter?",
];

type Props = {
  onSubmit: (q: string) => void;
  disabled?: boolean;
};

export function QueryInput({ onSubmit, disabled }: Props) {
  const [value, setValue] = useState("");

  const submit = (q?: string) => {
    const text = (q ?? value).trim();
    if (!text || disabled) return;
    onSubmit(text);
    if (!q) setValue("");
  };

  return (
    <div className="w-full">
      <div className="relative rounded-2xl border border-border bg-card transition-colors focus-within:border-mint/60">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Ask anything about your knowledge base…"
          rows={2}
          className="w-full resize-none bg-transparent px-5 py-4 pr-14 text-[15px] leading-relaxed text-foreground outline-none placeholder:text-muted-foreground"
        />
        <button
          onClick={() => submit()}
          disabled={disabled || !value.trim()}
          aria-label="Send query"
          className="absolute bottom-3 right-3 inline-flex h-9 w-9 items-center justify-center rounded-full bg-mint text-background transition disabled:cursor-not-allowed disabled:opacity-30 hover:opacity-90"
          style={{ backgroundColor: "var(--mint)" }}
        >
          <ArrowUp className="h-4 w-4" strokeWidth={2.5} />
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="mono-label inline-flex items-center gap-1.5">
          <Sparkles className="h-3 w-3" />
          Try
        </span>
        {SAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            onClick={() => submit(q)}
            disabled={disabled}
            className="rounded-full border border-border bg-card/50 px-3 py-1 text-xs text-muted-foreground transition hover:border-mint/40 hover:text-foreground disabled:opacity-40"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
