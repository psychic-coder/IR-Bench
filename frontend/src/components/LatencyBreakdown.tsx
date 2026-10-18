import type { Latency } from "@/lib/api";
import { fmtMs } from "@/lib/format";

const STAGES: { key: keyof Omit<Latency, "total">; label: string; color: string }[] = [
  { key: "extract", label: "extract", color: "bg-sky-500" },
  { key: "search", label: "search", color: "bg-violet-500" },
  { key: "rerank", label: "rerank", color: "bg-amber-500" },
];

/** Stacked bar + numbers for the per-stage latency of one /search call. */
export function LatencyBreakdown({
  latency,
  cache,
}: {
  latency: Latency;
  cache?: "hit" | "miss" | "disabled";
}) {
  const total = Math.max(latency.total, 1e-9);
  return (
    <div className="space-y-2" aria-label="Latency breakdown">
      <div className="flex h-3 w-full overflow-hidden rounded bg-slate-200 dark:bg-slate-800">
        {STAGES.map((s) => (
          <div
            key={s.key}
            className={s.color}
            style={{ width: `${(100 * latency[s.key]) / total}%` }}
            title={`${s.label}: ${fmtMs(latency[s.key])}`}
          />
        ))}
      </div>
      <dl className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
        {STAGES.map((s) => (
          <div key={s.key} className="flex items-center gap-1.5">
            <span className={`inline-block h-2 w-2 rounded-full ${s.color}`} aria-hidden />
            <dt className="text-slate-500">{s.label}</dt>
            <dd className="font-mono">{fmtMs(latency[s.key])}</dd>
          </div>
        ))}
        <div className="flex items-center gap-1.5">
          <dt className="font-medium">total</dt>
          <dd className="font-mono font-medium">{fmtMs(latency.total)}</dd>
        </div>
        {cache ? <CacheBadge cache={cache} /> : null}
      </dl>
    </div>
  );
}

export function CacheBadge({ cache }: { cache: "hit" | "miss" | "disabled" }) {
  const cls =
    cache === "hit"
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
      : cache === "miss"
        ? "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300"
        : "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300";
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${cls}`} data-testid="cache-badge">
      embedding cache {cache}
    </span>
  );
}
