"""Write results.json, results.csv, plots, summary.md and hard_queries.json."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

from irbench.bench.ablations import Table, all_tables, emh, p95
from irbench.bench.plots import make_plots
from irbench.bench.store import ResultsStore
from irbench.data.base import RetrievalDataset
from irbench.utils import atomic_write_json, ensure_dir, utc_now_iso

Record = dict[str, Any]

CSV_COLUMNS = [
    "run_id",
    "status",
    "dataset",
    "protocol",
    "family",
    "extractor",
    "pooling",
    "label",
    "index",
    "rerank",
    "n_distractors",
    "vector_dim",
    "headline_map",
    "map_easy",
    "map_medium",
    "map_hard",
    "map_original",
    "recall@1",
    "recall@5",
    "recall@10",
    "ann_recall@1",
    "ann_recall@k",
    "p50_extract_ms",
    "p95_extract_ms",
    "p99_extract_ms",
    "p50_search_ms",
    "p95_search_ms",
    "p99_search_ms",
    "p50_rerank_ms",
    "p95_rerank_ms",
    "p99_rerank_ms",
    "p50_total_ms",
    "p95_total_ms",
    "p99_total_ms",
    "qps_single_stream",
    "qps_search_batch",
    "bytes_per_vector",
    "encoding_bytes_per_vector",
    "index_total_bytes",
    "memory_method",
    "build_time_s",
    "extract_ms_per_image_db",
    "n_db",
    "n_queries",
    "config_hash",
    "experiments",
    "reason",
]


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if 0 < abs(value) < 0.01:
            return f"{value:.4f}"
        return f"{value:.{digits}f}"
    return str(value).replace("|", "/")


def markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    """GitHub-flavoured markdown table."""
    if not rows:
        return "_No runs for this table (not in the configured grid, or all runs skipped/failed)._\n"
    head = "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n"
    body = "".join("| " + " | ".join(_fmt(v) for v in row) + " |\n" for row in rows)
    return head + body


def csv_row(r: Record) -> dict[str, Any]:
    lat = r.get("latency_ms", {})
    mem = r.get("memory", {}) or {}
    med = (r.get("metrics") or {}).get("medium") or (r.get("metrics") or {}).get("all") or {}
    e, m, h = emh(r) if r.get("status") == "ok" else (None, None, None)
    k = r.get("ann_recall_k")
    row = {
        "run_id": r["run_id"],
        "status": r.get("status"),
        "dataset": r["dataset"],
        "protocol": r["protocol"],
        "family": r.get("family"),
        "extractor": r.get("extractor"),
        "pooling": r.get("pooling"),
        "label": r.get("label"),
        "index": r.get("index"),
        "rerank": r.get("rerank"),
        "n_distractors": r.get("n_distractors"),
        "vector_dim": r.get("vector_dim"),
        "headline_map": r.get("headline_map"),
        "map_easy": e if r["protocol"] == "revisited" else None,
        "map_medium": m if r["protocol"] == "revisited" else None,
        "map_hard": h if r["protocol"] == "revisited" else None,
        "map_original": m if r["protocol"] == "original" else None,
        "recall@1": med.get("recall@1"),
        "recall@5": med.get("recall@5"),
        "recall@10": med.get("recall@10"),
        "ann_recall@1": r.get("ann_recall@1"),
        "ann_recall@k": None if k is None else r.get(f"ann_recall@{k}"),
        "qps_single_stream": (r.get("throughput_qps") or {}).get("end_to_end_single_stream"),
        "qps_search_batch": (r.get("throughput_qps") or {}).get("search_batch"),
        "bytes_per_vector": mem.get("bytes_per_vector"),
        "encoding_bytes_per_vector": mem.get("encoding_bytes_per_vector"),
        "index_total_bytes": mem.get("total_bytes"),
        "memory_method": mem.get("method"),
        "build_time_s": r.get("build_time_s"),
        "extract_ms_per_image_db": r.get("extract_ms_per_image_db"),
        "n_db": r.get("n_db"),
        "n_queries": r.get("n_queries"),
        "config_hash": r.get("config_hash"),
        "experiments": ";".join(f"{x['ablation']}:{x['name']}" for x in r.get("experiments", [])),
        "reason": r.get("reason"),
    }
    for stage in ("extract", "search", "rerank", "total"):
        st = lat.get(stage) or {}
        for q in ("p50", "p95", "p99"):
            row[f"{q}_{stage}_ms"] = st.get(q)
    return row


def write_csv(records: list[Record], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(csv_row(r))


# ---------------------------------------------------------------------- hard queries
def hard_queries(
    records: list[Record],
    store: ResultsStore,
    get_dataset: Callable[[str], RetrievalDataset],
    threshold: float,
    top: int,
) -> dict[str, Any]:
    """Queries where the best configuration (per dataset) still has Medium AP < threshold."""
    out: dict[str, Any] = {"generated_at": utc_now_iso(), "threshold_ap": threshold, "datasets": {}}
    candidates = [
        r
        for r in records
        if r.get("status") == "ok"
        and r["protocol"] == "revisited"
        and r.get("n_distractors", 0) == 0
        and r.get("headline_map") is not None
    ]
    by_ds: dict[str, Record] = {}
    for r in candidates:
        if r["dataset"] not in by_ds or r["headline_map"] > by_ds[r["dataset"]]["headline_map"]:
            by_ds[r["dataset"]] = r
    for ds_name, best in sorted(by_ds.items()):
        ranks = store.load_rankings(best["run_id"])
        if ranks is None:
            continue
        ds = get_dataset(ds_name)
        proto = ds.protocols["revisited"]
        qindex = {q.id: i for i, q in enumerate(proto.queries)}
        items = []
        for qi_rec, qid in enumerate(best["query_ids"]):
            ap = best["per_query_ap"]["medium"][qi_rec]
            if ap is None or ap >= threshold or qid not in qindex:
                continue
            qi = qindex[qid]
            pos = {f"{ds_name}/{proto.db_ids[j]}" for j in proto.positives(qi, "medium")}
            junk = {f"{ds_name}/{proto.db_ids[j]}" for j in proto.junk(qi, "medium")}
            returned = []
            first_pos = None
            for rank, (iid, score) in enumerate(
                zip(ranks["top_ids"][qi_rec], ranks["top_scores"][qi_rec], strict=False)
            ):
                label = (
                    "positive"
                    if iid in pos
                    else "junk"
                    if iid in junk
                    else "distractor"
                    if iid.startswith("distractors/")
                    else "negative"
                )
                if label == "positive" and first_pos is None:
                    first_pos = rank + 1
                if rank < top:
                    returned.append({"rank": rank + 1, "id": iid, "score": score, "label": label})
            q = proto.queries[qi]
            items.append(
                {
                    "query_id": f"{ds_name}/{qid}",
                    "bbox": list(q.bbox) if q.bbox else None,
                    "ap": {
                        s: best["per_query_ap"][s][qi_rec]
                        for s in ("easy", "medium", "hard")
                        if s in best["per_query_ap"]
                    },
                    "n_positives": len(pos),
                    "n_junk": len(junk),
                    "expected_positives": sorted(pos)[:60],
                    "first_positive_rank_in_top50": first_pos,
                    "returned": returned,
                }
            )
        items.sort(key=lambda x: x["ap"].get("medium") or 0.0)
        out["datasets"][ds_name] = {
            "best_run_id": best["run_id"],
            "best_config": best["label"],
            "best_map_medium": best["headline_map"],
            "n_queries": len(best["query_ids"]),
            "n_hard": len(items),
            "queries": items,
        }
    return out


# ---------------------------------------------------------------------- summary
def summary_markdown(records: list[Record], meta: dict[str, Any], tables: list[Table]) -> str:
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    failed = [r for r in records if r.get("status") == "failed"]
    hw = meta.get("hardware", {})
    libs = meta.get("libraries", {})
    lines = [
        f"# Benchmark results: `{meta.get('benchmark')}`",
        "",
        f"_Generated {meta.get('generated_at')} by the irbench harness from `results/results.json`. "
        "Every number below is a measurement from this run; regenerate instead of editing._",
        "",
        "## Run metadata",
        "",
        markdown_table(
            ["key", "value"],
            [
                ["config", meta.get("benchmark")],
                ["config hash", meta.get("config_hash")],
                ["seed", meta.get("seed")],
                ["dataset subset (DATASET_SUBSET)", meta.get("subset") or "full"],
                ["device", meta.get("device")],
                ["CPU", f"{hw.get('cpu_model')} ({hw.get('cpu_logical_cores')} logical cores)"],
                ["GPU", hw.get("gpu_name", "none")],
                ["RAM (GB)", hw.get("ram_gb")],
                ["platform", hw.get("platform")],
                ["python", hw.get("python")],
                ["libraries", ", ".join(f"{k} {v}" for k, v in sorted(libs.items()))],
                ["runs ok / skipped / failed", f"{len(ok)} / {len(skipped)} / {len(failed)}"],
                ["wall time (s)", meta.get("wall_time_s")],
                [
                    "headline metric",
                    f"mAP, {meta.get('headline', {}).get('protocol')} protocol, "
                    f"{meta.get('headline', {}).get('setup')} setup",
                ],
            ],
        ),
        "",
        "## Datasets",
        "",
    ]
    ds_rows = []
    for name, d in sorted((meta.get("datasets") or {}).items()):
        protos = d.get("protocols", {})
        ds_rows.append(
            [
                name,
                d.get("subset") or "full",
                d.get("n_images_pool"),
                (protos.get("revisited") or {}).get("n_db"),
                (protos.get("revisited") or {}).get("n_queries"),
                (protos.get("original") or {}).get("n_db"),
                (protos.get("original") or {}).get("n_queries"),
                d.get("n_corrupt"),
                d.get("n_missing_referenced"),
            ]
        )
    lines += [
        markdown_table(
            [
                "dataset",
                "subset",
                "images",
                "revisited db",
                "revisited queries",
                "original db",
                "original queries",
                "corrupt (skipped)",
                "missing referenced",
            ],
            ds_rows,
        ),
        "",
    ]
    for name, d in sorted((meta.get("distractors") or {}).items()):
        lines.append(
            f"- Distractors ({name}): {d.get('available')} usable from `{d.get('source')}`, "
            f"{d.get('skipped_corrupt')} corrupt skipped. {' '.join(d.get('notes') or [])}"
        )
    lines += ["", "## Headline: top configurations per dataset", ""]
    head_rows = []
    hp = (meta.get("headline") or {}).get("protocol", "revisited")
    for ds in sorted({r["dataset"] for r in ok}):
        top = sorted(
            (
                r
                for r in ok
                if r["dataset"] == ds and r["protocol"] == hp and r.get("headline_map") is not None
            ),
            key=lambda r: -r["headline_map"],
        )[:10]
        for r in top:
            e, m, h = emh(r)
            med = r["metrics"].get("medium") or r["metrics"].get("all") or {}
            head_rows.append(
                [
                    ds,
                    r["label"],
                    r.get("family"),
                    e,
                    m,
                    h,
                    med.get("recall@1"),
                    p95(r, "total"),
                    r["memory"]["bytes_per_vector"],
                    r["memory"]["method"],
                ]
            )
    lines += [
        markdown_table(
            [
                "dataset",
                "config",
                "family",
                "mAP E",
                "mAP M",
                "mAP H",
                "R@1 M",
                "p95 total ms",
                "B/vec",
                "memory",
            ],
            head_rows,
        ),
        "",
    ]
    for t in tables:
        lines += [f"## ({t.key.split('_')[0]}) {t.title}", ""]
        if t.plot:
            lines += [f"![{t.title}]({t.plot})", ""]
        lines.append(markdown_table(t.columns, t.rows))
        for n in t.notes:
            lines.append(f"- {n}")
        lines.append("")
    if skipped or failed:
        lines += ["## Skipped and failed runs", ""]
        lines.append(
            markdown_table(
                ["status", "dataset", "protocol", "config", "reason"],
                [
                    [r["status"], r["dataset"], r["protocol"], r.get("label"), (r.get("reason") or "")[:160]]
                    for r in skipped + failed
                ],
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    records: list[Record],
    meta: dict[str, Any],
    results_dir: Path,
    get_dataset: Callable[[str], RetrievalDataset],
    hard_threshold: float,
    hard_top: int,
) -> dict[str, Path]:
    """Write every output artefact; returns their paths."""
    results_dir = ensure_dir(results_dir)
    headline_protocol = (meta.get("headline") or {}).get("protocol", "revisited")
    tables = all_tables(records, hard_threshold, headline_protocol)
    make_plots(records, tables, results_dir / "plots", headline_protocol)
    store = ResultsStore(results_dir)
    paths = {
        "json": results_dir / "results.json",
        "csv": results_dir / "results.csv",
        "summary": results_dir / "summary.md",
        "hard": results_dir / "hard_queries.json",
    }
    atomic_write_json(
        paths["json"],
        {
            "meta": meta,
            "runs": [{k: v for k, v in r.items() if k != "traceback"} for r in records],
            "ablations": [t.to_dict() for t in tables],
        },
    )
    write_csv(records, paths["csv"])
    paths["summary"].write_text(summary_markdown(records, meta, tables), encoding="utf-8")
    atomic_write_json(paths["hard"], hard_queries(records, store, get_dataset, hard_threshold, hard_top))
    return paths
