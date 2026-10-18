import type { ReactNode } from "react";

import type { ApiError, Ready } from "@/lib/api";

export function Panel({
  title,
  children,
  tone = "neutral",
}: {
  title: string;
  children?: ReactNode;
  tone?: "neutral" | "warn" | "error";
}) {
  const toneClass =
    tone === "error"
      ? "border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40"
      : tone === "warn"
        ? "border-amber-300 bg-amber-50 dark:border-amber-900 dark:bg-amber-950/40"
        : "border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900";
  return (
    <section className={`rounded-lg border p-5 ${toneClass}`} aria-live="polite">
      <h2 className="text-base font-semibold">{title}</h2>
      {children ? <div className="mt-2 text-sm text-slate-600 dark:text-slate-300">{children}</div> : null}
    </section>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-3 py-8 text-sm text-slate-500 dark:text-slate-400">
      <span
        className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-accent"
        aria-hidden
      />
      {label}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return <Panel title={title}>{children}</Panel>;
}

export function ErrorState({ error, onRetry }: { error: ApiError | Error; onRetry?: () => void }) {
  const code = "code" in error ? (error as ApiError).code : "error";
  return (
    <Panel title="Something went wrong" tone="error">
      <p>
        <span className="font-mono text-xs">{code}</span> — {error.message}
      </p>
      {onRetry ? (
        <button type="button" onClick={onRetry} className="btn-secondary mt-3">
          Retry
        </button>
      ) : null}
    </Panel>
  );
}

export function WarmingUp({ status }: { status: Ready | null }) {
  const progress = status?.progress;
  const pct =
    progress && typeof progress.done === "number" && typeof progress.total === "number" && progress.total > 0
      ? Math.round((100 * progress.done) / progress.total)
      : null;
  if (status?.stage === "error") {
    return (
      <Panel title="Backend could not finish warming up" tone="error">
        <p>{status.message}</p>
        {status.error ? (
          <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-slate-950 p-3 font-mono text-xs text-slate-100">
            {status.error}
          </pre>
        ) : null}
      </Panel>
    );
  }
  return (
    <Panel title="Backend warming up" tone="warn">
      <p>
        The API is loading the dataset, extracting database descriptors and building indexes. Search becomes
        available automatically when it is ready.
      </p>
      <p className="mt-2 font-mono text-xs">
        stage: {status?.stage ?? "connecting"} — {status?.message ?? "waiting for the API"}
      </p>
      {pct !== null ? (
        <div className="mt-3" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
          <div className="h-2 w-full overflow-hidden rounded bg-amber-200 dark:bg-amber-900">
            <div className="h-2 bg-amber-500" style={{ width: `${pct}%` }} />
          </div>
          <p className="mt-1 text-xs">
            {progress?.done} / {progress?.total} images ({pct}%)
          </p>
        </div>
      ) : null}
    </Panel>
  );
}
