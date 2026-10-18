/** Formatting and small pure helpers shared by pages (unit-tested). */

export function fmtNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return value.toFixed(digits);
}

export function fmtMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  if (value >= 100) return `${value.toFixed(0)} ms`;
  return `${value.toFixed(1)} ms`;
}

export function fmtBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  const units = ["B", "KB", "MB", "GB"];
  let v = value;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export interface Point {
  x: number;
  y: number;
}

/** Non-dominated points for "minimise x (latency), maximise y (mAP)", sorted by x. */
export function paretoFrontier<T extends Point>(points: T[]): T[] {
  const sorted = [...points].sort((a, b) => a.x - b.x || b.y - a.y);
  const front: T[] = [];
  let best = -Infinity;
  for (const p of sorted) {
    if (p.y > best) {
      front.push(p);
      best = p.y;
    }
  }
  return front;
}

export type SortDir = "asc" | "desc";

/** Stable sort with nulls last regardless of direction. */
export function sortRows<T>(
  rows: T[],
  get: (row: T) => number | string | null | undefined,
  dir: SortDir,
): T[] {
  return rows
    .map((row, i) => ({ row, i, v: get(row) }))
    .sort((a, b) => {
      const an = a.v === null || a.v === undefined;
      const bn = b.v === null || b.v === undefined;
      if (an && bn) return a.i - b.i;
      if (an) return 1;
      if (bn) return -1;
      const cmp =
        typeof a.v === "number" && typeof b.v === "number"
          ? a.v - b.v
          : String(a.v).localeCompare(String(b.v));
      return (dir === "asc" ? cmp : -cmp) || a.i - b.i;
    })
    .map((x) => x.row);
}

/** 1-2-5 ticks spanning [min, max] for a log axis; returns the padded domain too. */
export function logTicks(min: number, max: number): { ticks: number[]; domain: [number, number] } {
  if (!(min > 0) || !(max > 0)) return { ticks: [], domain: [1, 10] };
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const ticks: number[] = [];
  for (let e = lo; e <= hi; e += 1) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= min / 1.5 && v <= max * 1.5) ticks.push(Number(v.toPrecision(3)));
    }
  }
  const domain: [number, number] = [
    Math.min(min, ticks[0] ?? min) / 1.2,
    Math.max(max, ticks[ticks.length - 1] ?? max) * 1.2,
  ];
  return { ticks, domain };
}

/** Compact axis label: 1500 -> "1.5k", 2000000 -> "2M". */
export function fmtCompact(v: number): string {
  if (v >= 1e6) return `${Number((v / 1e6).toPrecision(3))}M`;
  if (v >= 1e3) return `${Number((v / 1e3).toPrecision(3))}k`;
  return `${Number(v.toPrecision(3))}`;
}
