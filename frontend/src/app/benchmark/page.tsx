"use client";

import { useMemo, useState } from "react";

import { BenchmarkCharts } from "@/components/BenchmarkCharts";
import { ReadinessGate } from "@/components/ReadinessGate";
import { SortableTable, type Column } from "@/components/SortableTable";
import { EmptyState, ErrorState, Loading } from "@/components/StateViews";
import { getBenchmark, type Run } from "@/lib/api";
import { fmtBytes, fmtMs, fmtNumber } from "@/lib/format";
import { useAsync } from "@/lib/hooks";

const mapOf = (r: Run, setup: string) => r.metrics?.[setup]?.map ?? null;

const COLUMNS: Column<Run>[] = [
  {
    key: "label",
    header: "Configuration",
    value: (r) => r.label,
    render: (r) => <span className="font-mono">{r.label}</span>,
  },
  { key: "dataset", header: "Dataset", value: (r) => r.dataset },
  { key: "protocol", header: "Protocol", value: (r) => r.protocol },
  { key: "family", header: "Family", value: (r) => r.family ?? null },
  {
    key: "headline",
    header: "mAP (headline)",
    value: (r) => r.headline_map ?? null,
    render: (r) => fmtNumber(r.headline_map),
    numeric: true,
  },
  {
    key: "easy",
    header: "E",
    value: (r) => mapOf(r, "easy"),
    render: (r) => fmtNumber(mapOf(r, "easy")),
    numeric: true,
  },
  {
    key: "medium",
    header: "M",
    value: (r) => mapOf(r, "medium") ?? mapOf(r, "all"),
    render: (r) => fmtNumber(mapOf(r, "medium") ?? mapOf(r, "all")),
    numeric: true,
  },
  {
    key: "hard",
    header: "H",
    value: (r) => mapOf(r, "hard"),
    render: (r) => fmtNumber(mapOf(r, "hard")),
    numeric: true,
  },
  {
    key: "r1",
    header: "R@1",
    value: (r) => (r.metrics?.medium ?? r.metrics?.all)?.["recall@1"] ?? null,
    render: (r) => fmtNumber((r.metrics?.medium ?? r.metrics?.all)?.["recall@1"] ?? null, 1),
    numeric: true,
  },
  {
    key: "p95",
    header: "p95 total",
    value: (r) => r.latency_ms?.total?.p95 ?? null,
    render: (r) => fmtMs(r.latency_ms?.total?.p95),
    numeric: true,
  },
  {
    key: "bpv",
    header: "B/vec",
    value: (r) => r.memory?.bytes_per_vector ?? null,
    render: (r) => fmtBytes(r.memory?.bytes_per_vector),
    numeric: true,
  },
  { key: "index", header: "Index", value: (r) => r.index },
  { key: "rerank", header: "Re-rank", value: (r) => r.rerank },
];

function BenchmarkView() {
  const bench = useAsync(getBenchmark);
  const [dataset, setDataset] = useState("all");
  const [protocol, setProtocol] = useState("revisited");

  const ok = useMemo(() => (bench.data?.runs ?? []).filter((r) => r.status === "ok"), [bench.data]);
  const datasets = useMemo(() => [...new Set(ok.map((r) => r.dataset))].sort(), [ok]);
  const protocols = useMemo(() => [...new Set(ok.map((r) => r.protocol))].sort(), [ok]);
  const filtered = useMemo(
    () =>
      ok.filter(
        (r) =>
          (dataset === "all" || r.dataset === dataset) && (protocol === "all" || r.protocol === protocol),
      ),
    [ok, dataset, protocol],
  );

  if (bench.loading) return <Loading label="Loading results.json…" />;
  if (bench.error?.code === "no_results") {
    return (
      <EmptyState title="No results generated yet">
        Run <code className="font-mono">make bench-fast</code> (smoke test) or{" "}
        <code className="font-mono">make bench</code> (full benchmark). Results appear here as soon as{" "}
        <code className="font-mono">results/results.json</code> exists.
      </EmptyState>
    );
  }
  if (bench.error) return <ErrorState error={bench.error} onRetry={bench.reload} />;
  if (!bench.data) return null;
  const meta = bench.data.meta;
  const hw = meta.hardware as Record<string, unknown>;
  const notOk = bench.data.runs.length - ok.length;

  return (
    <div className="space-y-6">
      <section className="card grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-4" aria-label="Run metadata">
        <p>
          <span className="text-slate-500">config</span> <span className="font-mono">{meta.benchmark}</span> (
          {meta.config_hash})
        </p>
        <p>
          <span className="text-slate-500">generated</span> {meta.generated_at}
        </p>
        <p>
          <span className="text-slate-500">device</span> {meta.device} · {String(hw.cpu_model ?? "")}
        </p>
        <p>
          <span className="text-slate-500">seed</span> {meta.seed} ·{" "}
          <span className="text-slate-500">subset</span> {meta.subset ?? "full"}
        </p>
        <p className="sm:col-span-2 lg:col-span-4">
          <span className="text-slate-500">headline</span> mAP, {meta.headline.protocol} protocol,{" "}
          {meta.headline.setup} setup · {ok.length} successful runs{notOk ? `, ${notOk} skipped/failed` : ""}
        </p>
      </section>

      <div className="flex flex-wrap items-end gap-4">
        <div>
          <label
            htmlFor="ds-filter"
            className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500"
          >
            Dataset
          </label>
          <select
            id="ds-filter"
            className="input"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
          >
            <option value="all">all</option>
            {datasets.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label
            htmlFor="proto-filter"
            className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500"
          >
            Protocol
          </label>
          <select
            id="proto-filter"
            className="input"
            value={protocol}
            onChange={(e) => setProtocol(e.target.value)}
          >
            <option value="all">all</option>
            {protocols.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
        <p className="text-xs text-slate-500">{filtered.length} runs shown</p>
      </div>

      {filtered.length === 0 ? (
        <EmptyState title="No runs match this filter" />
      ) : (
        <>
          <BenchmarkCharts runs={filtered} />
          <SortableTable
            caption="Benchmark results matrix"
            rows={filtered}
            columns={COLUMNS}
            rowKey={(r) => r.run_id}
            initialSort={{ key: "headline", dir: "desc" }}
          />
        </>
      )}

      <section className="space-y-3" aria-label="Ablation notes">
        <h2 className="text-sm font-semibold">Ablation findings (generated by the harness)</h2>
        {bench.data.ablations.map((t) => (
          <details key={t.key} className="card">
            <summary className="cursor-pointer text-sm font-medium">
              ({t.key.split("_")[0]}) {t.title}{" "}
              <span className="text-xs text-slate-500">· {t.rows.length} rows</span>
            </summary>
            {t.notes.length ? (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-slate-600 dark:text-slate-300">
                {t.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            ) : (
              <p className="mt-2 text-xs text-slate-500">No notes.</p>
            )}
          </details>
        ))}
      </section>
    </div>
  );
}

export default function BenchmarkPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Benchmark results</h1>
      <ReadinessGate blocking={false}>
        <BenchmarkView />
      </ReadinessGate>
    </div>
  );
}
