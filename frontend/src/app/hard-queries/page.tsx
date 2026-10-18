"use client";

import { useState } from "react";

import { ReadinessGate } from "@/components/ReadinessGate";
import { EmptyState, ErrorState, Loading } from "@/components/StateViews";
import { getHardQueries, imageUrl, type HardQueries } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import { useAsync } from "@/lib/hooks";

type HardQuery = HardQueries["datasets"][string]["queries"][number];

const LABEL_STYLE: Record<string, string> = {
  positive: "border-emerald-500 text-emerald-700 dark:text-emerald-400",
  junk: "border-amber-500 text-amber-700 dark:text-amber-400",
  negative: "border-red-500 text-red-700 dark:text-red-400",
  distractor: "border-slate-400 text-slate-600 dark:text-slate-400",
};

function Thumb({ id, caption, className = "" }: { id: string; caption?: string; className?: string }) {
  return (
    <figure className={`w-28 shrink-0 ${className}`}>
      <a href={imageUrl(id, "images")} target="_blank" rel="noreferrer">
        <img
          src={imageUrl(id, "thumbnails")}
          alt={caption ?? id}
          loading="lazy"
          className="aspect-[4/3] w-full rounded object-cover"
        />
      </a>
      {caption ? <figcaption className="mt-0.5 truncate text-[10px]">{caption}</figcaption> : null}
    </figure>
  );
}

function QueryCard({ q }: { q: HardQuery }) {
  return (
    <article className="card space-y-3">
      <header className="flex flex-wrap items-start gap-4">
        <Thumb id={q.query_id} caption="query (full image)" />
        <div className="space-y-1 text-xs">
          <h3 className="font-mono text-sm font-semibold">{q.query_id}</h3>
          <p>
            AP easy {fmtNumber(q.ap.easy ?? null, 3)} · medium {fmtNumber(q.ap.medium ?? null, 3)} · hard{" "}
            {fmtNumber(q.ap.hard ?? null, 3)}
          </p>
          <p>
            {q.n_positives} positives, {q.n_junk} junk · first positive in top-50:{" "}
            {q.first_positive_rank_in_top50 ?? "none"}
          </p>
          {q.bbox ? (
            <p className="font-mono text-slate-500">
              query box [{q.bbox.map((v) => v.toFixed(0)).join(", ")}]
            </p>
          ) : null}
        </div>
      </header>
      <div>
        <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Expected (positives, first 8)
        </h4>
        <div className="flex gap-2 overflow-x-auto pb-1">
          {q.expected_positives.slice(0, 8).map((id) => (
            <Thumb key={id} id={id} />
          ))}
        </div>
      </div>
      <div>
        <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Returned by the best configuration
        </h4>
        <ol className="flex gap-2 overflow-x-auto pb-1">
          {q.returned.map((r) => (
            <li key={r.rank} className={`rounded border-2 p-0.5 ${LABEL_STYLE[r.label] ?? ""}`}>
              <Thumb id={r.id} caption={`#${r.rank} ${r.label}`} />
            </li>
          ))}
        </ol>
      </div>
    </article>
  );
}

function HardQueriesView() {
  const hq = useAsync(getHardQueries);
  const [dataset, setDataset] = useState<string | null>(null);
  if (hq.loading) return <Loading label="Loading hard_queries.json…" />;
  if (hq.error?.code === "no_results") {
    return (
      <EmptyState title="No results generated yet">
        Hard queries are written by the benchmark (<code className="font-mono">make bench-fast</code>).
      </EmptyState>
    );
  }
  if (hq.error) return <ErrorState error={hq.error} onRetry={hq.reload} />;
  if (!hq.data) return null;
  const names = Object.keys(hq.data.datasets).sort();
  if (names.length === 0) return <EmptyState title="No hard queries recorded" />;
  const current = dataset ?? names[0]!;
  const ds = hq.data.datasets[current];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <label
            htmlFor="hq-ds"
            className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500"
          >
            Dataset
          </label>
          <select id="hq-ds" className="input" value={current} onChange={(e) => setDataset(e.target.value)}>
            {names.map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </div>
        {ds ? (
          <p className="text-xs text-slate-600 dark:text-slate-300">
            Best config <span className="font-mono">{ds.best_config}</span> (mAP Medium{" "}
            {fmtNumber(ds.best_map_medium)}) still fails on {ds.n_hard} of {ds.n_queries} queries (AP &lt;{" "}
            {hq.data.threshold_ap}).
          </p>
        ) : null}
      </div>
      <p className="flex flex-wrap gap-3 text-xs">
        {Object.keys(LABEL_STYLE).map((l) => (
          <span key={l} className={`rounded border-2 px-2 ${LABEL_STYLE[l]}`}>
            {l}
          </span>
        ))}
      </p>
      {ds && ds.queries.length === 0 ? (
        <EmptyState title="Every query is above the threshold for this dataset" />
      ) : null}
      {ds?.queries.map((q) => <QueryCard key={q.query_id} q={q} />)}
    </div>
  );
}

export default function HardQueriesPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Hard queries</h1>
      <ReadinessGate blocking={false}>
        <HardQueriesView />
      </ReadinessGate>
    </div>
  );
}
