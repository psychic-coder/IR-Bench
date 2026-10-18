"use client";

import type { ReactNode } from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import type { Run } from "@/lib/api";
import { fmtCompact, logTicks, paretoFrontier } from "@/lib/format";

const FAMILY_COLORS: Record<string, string> = {
  classical: "#64748b",
  cnn: "#2563eb",
  transformer: "#dc2626",
};
const SERIES_COLORS = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#9333ea",
  "#ea580c",
  "#0891b2",
  "#db2777",
  "#65a30d",
];
const LEGEND_STYLE = { fontSize: 11 };
const AXIS_TICK = { fontSize: 11 };

interface Pt {
  x: number;
  y: number;
  label: string;
}

function p95Total(r: Run): number | null {
  return r.latency_ms?.total?.p95 ?? null;
}

function annRecall(r: Run): number | null {
  const k = r.ann_recall_k;
  if (k === undefined) return null;
  const v = (r as Record<string, unknown>)[`ann_recall@${k}`];
  return typeof v === "number" ? v : null;
}

function ChartCard({ title, children, note }: { title: string; children: ReactNode; note?: string }) {
  return (
    <figure className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <figcaption className="mb-2 text-sm font-semibold">{title}</figcaption>
      <div className="h-72">{children}</div>
      {note ? <p className="mt-2 text-xs text-slate-500">{note}</p> : null}
    </figure>
  );
}

function NoData() {
  return (
    <p className="flex h-full items-center justify-center text-sm text-slate-500">
      No runs for this chart in the current filter.
    </p>
  );
}

function PointTooltip({ active, payload }: { active?: boolean; payload?: { payload: Pt }[] }) {
  const p = payload?.[0]?.payload;
  if (!active || !p) return null;
  return (
    <div className="max-w-xs rounded border border-slate-200 bg-white p-2 text-xs shadow dark:border-slate-700 dark:bg-slate-900">
      <p className="font-mono">{p.label}</p>
      <p>
        x = {fmtCompact(p.x)} · y = {p.y.toFixed(3)}
      </p>
    </div>
  );
}

function byFamily(runs: Run[], x: (r: Run) => number | null): Record<string, Pt[]> {
  const out: Record<string, Pt[]> = {};
  for (const r of runs) {
    const xv = x(r);
    if (xv === null || xv <= 0 || r.headline_map === null || r.headline_map === undefined) continue;
    const fam = r.family ?? "other";
    (out[fam] ??= []).push({ x: xv, y: r.headline_map, label: r.label });
  }
  return out;
}

function sweepSeries(runs: Run[], ablation: string, param: string): Record<string, Pt[]> {
  const out: Record<string, Pt[]> = {};
  for (const r of runs) {
    if (!r.experiments.some((e) => e.ablation === ablation)) continue;
    const pv = r.index_params_effective?.[param];
    const rec = annRecall(r);
    if (typeof pv !== "number" || rec === null) continue;
    const params = Object.entries(r.index_params_effective ?? {})
      .filter(([k]) => ![param, "seed", "n_train"].includes(k))
      .map(([k, v]) => `${k}=${String(v)}`)
      .join(",");
    const key = `${r.dataset} ${r.extractor}/${r.pooling ?? ""} ${params}`;
    (out[key] ??= []).push({ x: pv, y: rec, label: r.label });
  }
  for (const k of Object.keys(out)) out[k]!.sort((a, b) => a.x - b.x);
  return out;
}

function xRange(data: Record<string, Pt[]>): [number, number] {
  const xs = Object.values(data)
    .flat()
    .map((p) => p.x);
  return [Math.min(...xs), Math.max(...xs)];
}

function FamilyScatter({
  data,
  xLabel,
  frontier,
}: {
  data: Record<string, Pt[]>;
  xLabel: string;
  frontier?: Pt[];
}) {
  if (Object.keys(data).length === 0) return <NoData />;
  const [lo, hi] = xRange(data);
  const { ticks, domain } = logTicks(lo, hi);
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
        <CartesianGrid strokeOpacity={0.3} />
        <XAxis
          type="number"
          dataKey="x"
          scale="log"
          domain={domain}
          ticks={ticks}
          allowDataOverflow
          tick={AXIS_TICK}
          tickFormatter={fmtCompact}
          label={{ value: xLabel, position: "insideBottom", offset: -12, fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={[0, "auto"]}
          tick={AXIS_TICK}
          label={{ value: "mAP", angle: -90, position: "insideLeft", fontSize: 11 }}
        />
        <ZAxis range={[40, 40]} />
        <Tooltip content={<PointTooltip />} />
        <Legend verticalAlign="top" height={24} wrapperStyle={LEGEND_STYLE} />
        {Object.entries(data).map(([fam, pts]) => (
          <Scatter key={fam} name={fam} data={pts} fill={FAMILY_COLORS[fam] ?? "#0f766e"} />
        ))}
        {frontier && frontier.length > 0 ? (
          <Scatter
            name="Pareto frontier"
            data={frontier}
            fill="#0f172a"
            line={{ stroke: "#94a3b8", strokeDasharray: "4 3" }}
            lineType="joint"
            shape="diamond"
          />
        ) : null}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function SweepChart({ data, xLabel }: { data: Record<string, Pt[]>; xLabel: string }) {
  if (Object.keys(data).length === 0) return <NoData />;
  const ticks = [...new Set(Object.values(data).flatMap((pts) => pts.map((p) => p.x)))].sort((a, b) => a - b);
  const domain: [number, number] = [(ticks[0] ?? 1) / 1.3, (ticks[ticks.length - 1] ?? 10) * 1.3];
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
        <CartesianGrid strokeOpacity={0.3} />
        <XAxis
          type="number"
          dataKey="x"
          scale="log"
          domain={domain}
          ticks={ticks}
          allowDataOverflow
          tick={AXIS_TICK}
          tickFormatter={fmtCompact}
          label={{ value: xLabel, position: "insideBottom", offset: -12, fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={[0, 1]}
          tick={AXIS_TICK}
          label={{ value: "neighbour recall", angle: -90, position: "insideLeft", fontSize: 11 }}
        />
        <ZAxis range={[30, 30]} />
        <Tooltip content={<PointTooltip />} />
        <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 10 }} />
        {Object.entries(data).map(([name, pts], i) => (
          <Scatter
            key={name}
            name={name}
            data={pts}
            fill={SERIES_COLORS[i % SERIES_COLORS.length]}
            line
            lineType="joint"
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** The four benchmark charts, computed from the (already filtered) successful runs. */
export function BenchmarkCharts({ runs }: { runs: Run[] }) {
  const memory = byFamily(runs, (r) => r.memory?.bytes_per_vector ?? null);
  const latency = byFamily(runs, p95Total);
  const frontier = paretoFrontier(Object.values(latency).flat());
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <ChartCard
        title="mAP vs memory per vector"
        note="Bytes per vector amortise the whole index (codes, ids, codebooks, graph links)."
      >
        <FamilyScatter data={memory} xLabel="bytes / vector (log)" />
      </ChartCard>
      <ChartCard
        title="mAP vs p95 end-to-end latency (Pareto)"
        note="Latency = extract + search + rerank, warm-up excluded."
      >
        <FamilyScatter data={latency} xLabel="p95 latency, ms (log)" frontier={frontier} />
      </ChartCard>
      <ChartCard title="HNSW neighbour recall vs ef" note="Overlap of HNSW top-k with exact top-k.">
        <SweepChart data={sweepSeries(runs, "c", "ef")} xLabel="ef (log)" />
      </ChartCard>
      <ChartCard title="IVF-PQ neighbour recall vs nprobe" note="Overlap of IVF-PQ top-k with exact top-k.">
        <SweepChart data={sweepSeries(runs, "d", "nprobe")} xLabel="nprobe (log)" />
      </ChartCard>
    </div>
  );
}
