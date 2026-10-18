import type { SearchResult } from "@/lib/api";
import { imageUrl } from "@/lib/api";
import { fmtNumber } from "@/lib/format";

/** Thumbnail grid of ranked results; `highlight` marks ids unique to this result set. */
export function ResultGrid({ results, highlight }: { results: SearchResult[]; highlight?: Set<string> }) {
  if (results.length === 0) {
    return <p className="text-sm text-slate-500">No results.</p>;
  }
  return (
    <ol className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
      {results.map((r) => {
        const unique = highlight?.has(r.id);
        return (
          <li
            key={`${r.rank}-${r.id}`}
            className={`overflow-hidden rounded-lg border bg-white dark:bg-slate-900 ${
              unique
                ? "border-fuchsia-500 ring-2 ring-fuchsia-400/60"
                : "border-slate-200 dark:border-slate-800"
            }`}
          >
            <a
              href={imageUrl(r.id, "images")}
              target="_blank"
              rel="noreferrer"
              className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <img
                src={imageUrl(r.id, "thumbnails")}
                alt={`Rank ${r.rank}: ${r.id}`}
                loading="lazy"
                className="aspect-[4/3] w-full bg-slate-100 object-cover dark:bg-slate-800"
              />
            </a>
            <div className="space-y-0.5 p-2 text-xs">
              <div className="flex justify-between">
                <span className="font-semibold">#{r.rank}</span>
                <span className="font-mono">{fmtNumber(r.score, 3)}</span>
              </div>
              <p className="truncate font-mono text-slate-500" title={r.id}>
                {r.id.split("/").slice(1).join("/")}
              </p>
              {r.inliers !== null && r.inliers !== undefined ? (
                <p className="text-slate-500">{r.inliers} RANSAC inliers</p>
              ) : null}
              {unique ? (
                <p className="font-medium text-fuchsia-600 dark:text-fuchsia-400">only in this config</p>
              ) : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
