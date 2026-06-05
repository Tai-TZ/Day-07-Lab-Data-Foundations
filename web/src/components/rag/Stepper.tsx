import { Check } from "lucide-react";

export type StepStatus = "idle" | "active" | "done";
export type Step = { id: string; label: string; sublabel: string; status: StepStatus };

export function Stepper({ steps }: { steps: Step[] }) {
  return (
    <div className="flex items-stretch gap-3">
      {steps.map((s, i) => (
        <div key={s.id} className="flex flex-1 items-center gap-3">
          <div className="flex flex-1 items-center gap-3 rounded-lg border border-border bg-card px-4 py-3">
            <StatusDot status={s.status} index={i} />
            <div className="min-w-0">
              <div className="mono-label leading-none">Step 0{i + 1}</div>
              <div className="mt-1 truncate text-sm text-foreground">{s.label}</div>
            </div>
          </div>
          {i < steps.length - 1 && (
            <div className="hidden h-px w-6 bg-border md:block" />
          )}
        </div>
      ))}
    </div>
  );
}

function StatusDot({ status, index }: { status: StepStatus; index: number }) {
  if (status === "done") {
    return (
      <div
        className="flex h-7 w-7 items-center justify-center rounded-full"
        style={{ backgroundColor: "color-mix(in oklab, var(--mint) 25%, transparent)" }}
      >
        <Check className="h-3.5 w-3.5" style={{ color: "var(--mint)" }} strokeWidth={3} />
      </div>
    );
  }
  if (status === "active") {
    return (
      <div className="flex h-7 w-7 items-center justify-center">
        <span
          className="pulse-dot block h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: "var(--mint)" }}
        />
      </div>
    );
  }
  return (
    <div className="flex h-7 w-7 items-center justify-center rounded-full border border-border text-xs text-muted-foreground">
      {index + 1}
    </div>
  );
}
