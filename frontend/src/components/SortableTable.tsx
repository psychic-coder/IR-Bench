"use client";

import { useMemo, useState, type ReactNode } from "react";

import { sortRows, type SortDir } from "@/lib/format";

export interface Column<T> {
  key: string;
  header: string;
  value: (row: T) => number | string | null | undefined;
  render?: (row: T) => ReactNode;
  numeric?: boolean;
}

/** Accessible table whose headers are buttons that toggle the sort column/direction. */
export function SortableTable<T>({
  rows,
  columns,
  initialSort,
  rowKey,
  caption,
}: {
  rows: T[];
  columns: Column<T>[];
  initialSort?: { key: string; dir: SortDir };
  rowKey: (row: T) => string;
  caption?: string;
}) {
  const [sort, setSort] = useState(initialSort ?? { key: columns[0]?.key ?? "", dir: "desc" as SortDir });
  const sorted = useMemo(() => {
    const col = columns.find((c) => c.key === sort.key);
    return col ? sortRows(rows, col.value, sort.dir) : rows;
  }, [rows, columns, sort]);

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
      <table className="min-w-full text-left text-xs">
        {caption ? <caption className="sr-only">{caption}</caption> : null}
        <thead className="bg-slate-50 dark:bg-slate-900">
          <tr>
            {columns.map((c) => {
              const active = sort.key === c.key;
              return (
                <th
                  key={c.key}
                  scope="col"
                  aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                  className={`whitespace-nowrap px-3 py-2 font-semibold ${c.numeric ? "text-right" : ""}`}
                >
                  <button
                    type="button"
                    className="rounded hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                    onClick={() =>
                      setSort((s) => ({
                        key: c.key,
                        dir: s.key === c.key && s.dir === "desc" ? "asc" : "desc",
                      }))
                    }
                  >
                    {c.header}
                    <span aria-hidden className="ml-1 text-slate-400">
                      {active ? (sort.dir === "asc" ? "▲" : "▼") : "↕"}
                    </span>
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
          {sorted.map((row) => (
            <tr key={rowKey(row)} className="hover:bg-slate-50 dark:hover:bg-slate-900/60">
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={`whitespace-nowrap px-3 py-1.5 ${c.numeric ? "text-right font-mono" : ""}`}
                >
                  {c.render ? c.render(row) : (c.value(row) ?? "–")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
