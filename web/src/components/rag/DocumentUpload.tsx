import { Upload } from "lucide-react";
import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { uploadMarkdown } from "@/lib/rag-api";

type Props = {
  onUploaded?: (docId: string) => void;
};

export function DocumentUpload({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  const mutation = useMutation({
    mutationFn: uploadMarkdown,
    onSuccess: (data) => {
      setStatus({ type: "ok", text: data.message });
      queryClient.invalidateQueries({ queryKey: ["rag-documents"] });
      onUploaded?.(data.document.id);
    },
    onError: (err: Error) => {
      setStatus({ type: "err", text: err.message });
    },
  });

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    setStatus(null);
    mutation.mutate(file);
  };

  return (
    <div className="border-b border-border p-6">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <span className="mono-label">Upload knowledge</span>
          <p className="mt-1 text-sm text-muted-foreground">
            Add your own <code className="text-foreground">.md</code> file to the vector store.
          </p>
        </div>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={mutation.isPending}
          className="inline-flex items-center gap-2 rounded-full border border-border px-4 py-2 text-sm transition hover:border-mint/40 disabled:opacity-50"
        >
          <Upload className="h-4 w-4" />
          {mutation.isPending ? "Uploading…" : "Choose file"}
        </button>
      </div>

      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        className="rounded-lg border border-dashed p-6 text-center transition-colors"
        style={{
          borderColor: dragOver ? "var(--mint)" : "var(--border)",
          backgroundColor: dragOver
            ? "color-mix(in oklab, var(--mint) 6%, transparent)"
            : "color-mix(in oklab, var(--background) 40%, transparent)",
        }}
      >
        <Upload className="mx-auto h-5 w-5 text-muted-foreground" />
        <p className="mt-2 text-sm text-foreground">Drag & drop a Markdown file here</p>
        <p className="mono-label mt-1">.md only · UTF-8 · max 5 MB</p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".md,text/markdown"
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {status && (
        <p
          className="mt-3 text-sm"
          style={{ color: status.type === "ok" ? "var(--mint)" : "hsl(var(--destructive))" }}
        >
          {status.text}
        </p>
      )}
    </div>
  );
}
