"""Turn benchmark records into one table per ablation (a-j).

Every number in these tables is read from run records produced by the harness; nothing
is estimated or typed in. Rows whose inputs are missing are omitted rather than filled.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from irbench.utils import stable_hash

Record = dict[str, Any]


@dataclass
class Table:
    """A rendered ablation: column names, raw row values and free-text notes."""

    key: str
    title: str
    columns: list[str]
    rows: list[list[Any]]
    notes: list[str] = field(default_factory=list)
    plot: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "columns": self.columns,
            "rows": self.rows,
            "notes": self.notes,
            "plot": self.plot,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------- accessors
def in_ablation(records: Iterable[Record], letter: str) -> list[Record]:
    return [
        r
        for r in records
        if r.get("status") == "ok" and any(e["ablation"] == letter for e in r.get("experiments", []))
    ]


def map_of(r: Record, setup: str) -> float | None:
    m = r.get("metrics", {}).get(setup)
    return None if m is None else m.get("map")


def emh(r: Record) -> tuple[float | None, float | None, float | None]:
    if r["protocol"] == "original":
        v = map_of(r, "all")
        return v, v, v
    return map_of(r, "easy"), map_of(r, "medium"), map_of(r, "hard")


def p95(r: Record, stage: str) -> float | None:
    if stage == "ann_search":
        return (r.get("ann_search_ms") or {}).get("p95")
    return r.get("latency_ms", {}).get(stage, {}).get("p95")


def ann_k(r: Record) -> float | None:
    k = r.get("ann_recall_k")
    return None if k is None else r.get(f"ann_recall@{k}")


def space_key(r: Record, drop_distractors: bool = False) -> str:
    """Identity of the descriptor space (extractor, pooling, post-processing, distractors)."""
    c = r["config"]
    return stable_hash(
        {
            "e": c["extractor"],
            "p": c["pooling"],
            "pp": c["postprocess"],
            "d": None if drop_distractors else c["distractors"],
            "ds": r["dataset"],
            "pr": r["protocol"],
        }
    )


def feature_label(r: Record) -> str:
    c = r["config"]
    pooling = r.get("pooling") or c["pooling"]["name"]
    lab = f"{c['extractor']['name']}/{pooling}"
    if pooling and "gem" in pooling:
        lab += f"(p={c['pooling']['p']:g})"
    if c["extractor"]["scales"] != [1.0]:
        lab += " ms"
    if c["postprocess"]["pca"] != "none":
        lab += f" {c['postprocess']['pca']}{c['postprocess']['dim'] or 'full'}"
    return lab


def group_by(records: Iterable[Record], key: Callable[[Record], Any]) -> dict[Any, list[Record]]:
    out: dict[Any, list[Record]] = defaultdict(list)
    for r in records:
        out[key(r)].append(r)
    return dict(out)


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


# ---------------------------------------------------------------------- (a) IVF-PQ memory
def ablation_a(records: list[Record]) -> Table:
    recs = in_ablation(records, "a")
    rows: list[list[Any]] = []
    slopes: list[float] = []
    for _, grp in sorted(group_by(recs, space_key).items()):
        flat = next((r for r in grp if r["index"] == "flat"), None)
        if flat is None:
            continue
        f_map = flat["headline_map"]
        f_enc = flat["memory"]["encoding_bytes_per_vector"]
        rows.append(
            [
                flat["dataset"],
                feature_label(flat),
                "flat",
                "-",
                "-",
                f_enc,
                flat["memory"]["bytes_per_vector"],
                1.0,
                f_map,
                0.0,
                None,
            ]
        )
        for r in sorted(
            (r for r in grp if r["index"] == "ivfpq"), key=lambda r: -r["memory"]["encoding_bytes_per_vector"]
        ):
            eff = r["index_params_effective"]
            enc = r["memory"]["encoding_bytes_per_vector"]
            ratio = f_enc / enc if enc else None
            loss = _delta(f_map, r["headline_map"])
            per_decade = (loss / math.log10(ratio)) if (loss is not None and ratio and ratio > 1) else None
            if per_decade is not None:
                slopes.append(per_decade)
            rows.append(
                [
                    r["dataset"],
                    feature_label(r),
                    "ivfpq",
                    eff.get("m"),
                    eff.get("nbits"),
                    enc,
                    r["memory"]["bytes_per_vector"],
                    ratio,
                    r["headline_map"],
                    loss,
                    per_decade,
                ]
            )
    notes = [
        "compression = flat encoding bytes / PQ code bytes per vector (ids, centroids and codebooks "
        "excluded); 'total B/vec' amortises the whole index including those overheads.",
        "mAP-loss-per-10x = (flat mAP - IVF-PQ mAP) / log10(compression).",
    ]
    if slopes:
        notes.append(
            f"Mean mAP points lost per 10x compression over {len(slopes)} IVF-PQ configs: "
            f"{float(np.mean(slopes)):.2f} (median {float(np.median(slopes)):.2f})."
        )
    return Table(
        "a",
        "IVF-PQ: mAP lost per 10x memory saved",
        [
            "dataset",
            "features",
            "index",
            "m",
            "nbits",
            "enc B/vec",
            "total B/vec",
            "compression x",
            "mAP",
            "mAP loss",
            "loss per 10x",
        ],
        rows,
        notes,
        extra={"loss_per_decade": slopes},
    )


# ---------------------------------------------------------------------- (b) pooling
def ablation_b(records: list[Record]) -> tuple[Table, Table]:
    recs = in_ablation(records, "b")
    rows: list[list[Any]] = []
    for (ds, ext), grp in sorted(group_by(recs, lambda r: (r["dataset"], r["extractor"])).items()):

        def best(pool: str, _grp: list[Record] = grp) -> float | None:
            cands = [
                r["headline_map"]
                for r in _grp
                if r["pooling"] == pool and (r["config"]["pooling"]["p"] == 3.0 or "gem" not in pool)
            ]
            return max(cands) if cands else None

        family = grp[0].get("family")
        if family == "transformer":
            a, g, x = best("patch_gap"), best("patch_gem"), best("cls")
            rows.append([ds, ext, "patch-mean", a, "patch-GeM", g, "CLS", x, _delta(g, a)])
        else:
            a, g, x = best("gap"), best("gem"), best("rmac")
            rows.append([ds, ext, "GAP", a, "GeM", g, "R-MAC", x, _delta(g, a)])
    pooling = Table(
        "b",
        "GeM vs GAP (and R-MAC / CLS) per backbone",
        ["dataset", "backbone", "pool A", "mAP A", "pool B", "mAP B", "pool C", "mAP C", "GeM - mean gain"],
        rows,
        [
            "GeM rows use p=3; transformer rows compare patch-token GeM vs patch-token mean, "
            "with CLS as reference."
        ],
    )
    sweep_rows: list[list[Any]] = []
    for r in sorted(recs, key=lambda r: (r["dataset"], r["extractor"], r["config"]["pooling"]["p"])):
        if r["pooling"] in {"gem", "patch_gem"}:
            sweep_rows.append(
                [r["dataset"], r["extractor"], r["pooling"], r["config"]["pooling"]["p"], r["headline_map"]]
            )
    sweep = Table("b_sweep", "GeM p sweep", ["dataset", "backbone", "pooling", "p", "mAP"], sweep_rows)
    return pooling, sweep


# ---------------------------------------------------------------------- (c)/(d) ANN sweeps
def _ann_sweep(records: list[Record], letter: str, kind: str, param: str, title: str) -> Table:
    recs = [r for r in in_ablation(records, letter) if r["index"] == kind]
    rows: list[list[Any]] = []
    notes: list[str] = []

    def build_key(r: Record) -> tuple:
        bp = {k: v for k, v in r["config"]["index"]["params"].items() if k != param}
        return (space_key(r), stable_hash(bp))

    for _, grp in sorted(group_by(recs, build_key).items()):
        grp = sorted(grp, key=lambda r: r["index_params_effective"].get(param, 0))
        for r in grp:
            rows.append(
                [
                    r["dataset"],
                    feature_label(r),
                    ",".join(
                        f"{k}={v}"
                        for k, v in sorted(r["index_params_effective"].items())
                        if k not in {param, "seed", "n_train"}
                    ),
                    r["index_params_effective"].get(param),
                    r.get("ann_recall@1"),
                    ann_k(r),
                    r["headline_map"],
                    p95(r, "ann_search"),
                    p95(r, "search"),
                ]
            )
        hi = [r for r in grp if (ann_k(r) or 0) >= 0.95]
        lo = [r for r in grp if ann_k(r) is not None and ann_k(r) < 0.9]
        lab = f"{grp[0]['dataset']} {feature_label(grp[0])}"
        if hi:
            notes.append(
                f"{lab}: smallest {param} reaching neighbour recall >= 0.95 is "
                f"{hi[0]['index_params_effective'].get(param)}."
            )
        else:
            notes.append(f"{lab}: no {param} value in the sweep reached neighbour recall >= 0.95.")
        if lo:
            notes.append(
                f"{lab}: recall falls below 0.90 at {param} <= "
                f"{max(r['index_params_effective'].get(param) for r in lo)}."
            )
    k = next((r.get("ann_recall_k") for r in recs if r.get("ann_recall_k")), 10)
    return Table(
        letter,
        title,
        [
            "dataset",
            "features",
            "build params",
            param,
            "ANN R@1",
            f"ANN R@{k}",
            "mAP",
            f"p95 top-{k} ms",
            "p95 depth-search ms",
        ],
        rows,
        [
            *notes,
            f"ANN R@k = overlap of the index's top-k with exact (flat) top-k at k={k}. mAP uses a depth "
            "search (eval_depth), for which Qdrant uses max(ef, limit), so mAP is insensitive to small ef.",
        ],
    )


def ablation_c(records: list[Record]) -> Table:
    return _ann_sweep(records, "c", "hnsw", "ef", "HNSW (Qdrant) neighbour recall vs ef")


def ablation_d(records: list[Record]) -> Table:
    return _ann_sweep(records, "d", "ivfpq", "nprobe", "IVF-PQ (FAISS) neighbour recall vs nprobe")


# ---------------------------------------------------------------------- (e) Pareto
def pareto_front(points: list[tuple[float, float, Any]]) -> list[tuple[float, float, Any]]:
    """Non-dominated points for (minimise latency x, maximise mAP y)."""
    front: list[tuple[float, float, Any]] = []
    best_y = -math.inf
    for x, y, payload in sorted(points, key=lambda t: (t[0], -t[1])):
        if y > best_y:
            front.append((x, y, payload))
            best_y = y
    return front


def ablation_e(records: list[Record], protocol: str = "revisited") -> Table:
    recs = [r for r in records if r.get("status") == "ok" and r["protocol"] == protocol]
    rows: list[list[Any]] = []
    fronts: dict[str, list[str]] = {}
    for ds, grp in sorted(group_by(recs, lambda r: r["dataset"]).items()):
        pts = [
            (p95(r, "total"), r["headline_map"], r)
            for r in grp
            if p95(r, "total") is not None and r["headline_map"] is not None
        ]
        front = pareto_front(pts)
        fronts[ds] = [r["run_id"] for _, _, r in front]
        for x, y, r in front:
            rows.append([ds, r["label"], r.get("family"), y, x, r["memory"]["bytes_per_vector"]])
    return Table(
        "e",
        "mAP vs p95 end-to-end latency: Pareto frontier",
        ["dataset", "config", "family", "mAP", "p95 total ms", "B/vec"],
        rows,
        [
            f"Frontier over all {len(recs)} successful {protocol}-protocol runs; "
            "total = extract + search + rerank."
        ],
        extra={"front_run_ids": fronts},
    )


# ---------------------------------------------------------------------- (f) re-ranking
def ablation_f(records: list[Record]) -> Table:
    recs = in_ablation(records, "f")
    rows: list[list[Any]] = []

    def key(r: Record) -> tuple:
        return (space_key(r), r["index"], stable_hash(r["config"]["index"]["params"]))

    for _, grp in sorted(group_by(recs, key).items()):
        base = next((r for r in grp if r["rerank"] == "none"), None)
        for r in sorted(grp, key=lambda r: (r["rerank"] != "none", r["rerank"])):
            e, m, h = emh(r)
            rows.append(
                [
                    r["dataset"],
                    feature_label(r),
                    r["rerank"],
                    e,
                    m,
                    h,
                    _delta(m, emh(base)[1]) if base else None,
                    (r["latency_ms"]["rerank"] or {}).get("p50"),
                    p95(r, "rerank"),
                    p95(r, "total"),
                ]
            )
    return Table(
        "f",
        "Re-ranking: AQE, alpha-QE and spatial verification",
        [
            "dataset",
            "features",
            "rerank",
            "mAP E",
            "mAP M",
            "mAP H",
            "dM vs none",
            "p50 rerank ms",
            "p95 rerank ms",
            "p95 total ms",
        ],
        rows,
        [
            "Spatial verification latency assumes database ORB features are precomputed (offline); "
            "query ORB, matching and RANSAC are timed."
        ],
    )


# ---------------------------------------------------------------------- (g) PCA-whitening
def ablation_g(records: list[Record]) -> Table:
    recs = in_ablation(records, "g")
    rows: list[list[Any]] = []

    def key(r: Record) -> tuple:
        c = r["config"]
        return (
            r["dataset"],
            r["protocol"],
            stable_hash(c["extractor"]),
            stable_hash(c["pooling"]),
            c["distractors"],
        )

    for _, grp in sorted(group_by(recs, key).items()):
        base = next((r for r in grp if r["config"]["postprocess"]["pca"] == "none"), None)
        for r in sorted(grp, key=lambda r: (r["config"]["postprocess"]["pca"], r["vector_dim"])):
            e, m, h = emh(r)
            pca = r.get("pca") or {}
            rows.append(
                [
                    r["dataset"],
                    f"{r['extractor']}/{r['pooling']}",
                    r["config"]["postprocess"]["pca"],
                    r["config"]["postprocess"]["dim"] or "full",
                    r["vector_dim"],
                    pca.get("fit_on"),
                    pca.get("explained_variance_ratio"),
                    e,
                    m,
                    h,
                    _delta(m, emh(base)[1]) if base else None,
                ]
            )
    return Table(
        "g",
        "PCA / PCA-whitening (fit on the held-out dataset) and output dimension",
        [
            "dataset",
            "features",
            "mode",
            "dim req",
            "dim",
            "fit on",
            "var kept",
            "mAP E",
            "mAP M",
            "mAP H",
            "dM vs none",
        ],
        rows,
        [
            "PCA is always fit on the other dataset's database (Paris for Oxford and vice versa), never on "
            "evaluation images or queries. 'dim' can be below 'dim req' when the training set rank "
            "is smaller."
        ],
    )


# ---------------------------------------------------------------------- (h) scale
def ablation_h(records: list[Record]) -> Table:
    recs = in_ablation(records, "h")
    rows: list[list[Any]] = []
    for _, grp in sorted(group_by(recs, lambda r: (space_key(r, drop_distractors=True), r["index"])).items()):
        base = next((r for r in grp if r["n_distractors"] == 0), None)
        for r in sorted(grp, key=lambda r: r["n_distractors"]):
            e, m, h = emh(r)
            rows.append(
                [
                    r["dataset"],
                    feature_label(r),
                    r["index"],
                    r["n_distractors"],
                    r["n_index"],
                    e,
                    m,
                    h,
                    _delta(m, emh(base)[1]) if base else None,
                    (r["metrics"].get("medium") or r["metrics"].get("all") or {}).get("recall@1"),
                    p95(r, "search"),
                    r["memory"]["total_bytes"],
                ]
            )
    return Table(
        "h",
        "Scale test: degradation as distractors are added",
        [
            "dataset",
            "features",
            "index",
            "distractors",
            "index size",
            "mAP E",
            "mAP M",
            "mAP H",
            "dM vs 0",
            "R@1 M",
            "p95 search ms",
            "index bytes",
        ],
        rows,
    )


# ---------------------------------------------------------------------- (i) families
def ablation_i(records: list[Record], threshold: float) -> Table:
    recs = in_ablation(records, "i")
    rows: list[list[Any]] = []
    analysis: dict[str, Any] = {}
    original = {(r["dataset"], r["config_hash"]): r for r in recs if r["protocol"] == "original"}
    for ds, grp in sorted(
        group_by([r for r in recs if r["protocol"] == "revisited"], lambda r: r["dataset"]).items()
    ):
        best: dict[str, Record] = {}
        for r in grp:
            fam = r.get("family", "?")
            if r["headline_map"] is not None and (
                fam not in best or r["headline_map"] > best[fam]["headline_map"]
            ):
                best[fam] = r
        for fam in sorted(best):
            r = best[fam]
            e, m, h = emh(r)
            orig = original.get((ds, r["config_hash"]))
            rows.append(
                [
                    ds,
                    fam,
                    feature_label(r),
                    e,
                    m,
                    h,
                    orig["headline_map"] if orig else None,
                    p95(r, "total"),
                    r["extract_ms_per_image_db"],
                    r["memory"]["bytes_per_vector"],
                ]
            )
        # failure analysis on medium-protocol per-query AP of each family's best config
        qids = grp[0]["query_ids"]
        per_fam = {
            f: np.array([np.nan if a is None else a for a in r["per_query_ap"]["medium"]])
            for f, r in best.items()
        }
        fam_info: dict[str, Any] = {}
        for fam, aps in per_fam.items():
            fails = [i for i, a in enumerate(aps) if not np.isnan(a) and a < threshold]
            others = [f for f in per_fam if f != fam]
            unique = [
                i
                for i in fails
                if others
                and max((per_fam[o][i] for o in others if not np.isnan(per_fam[o][i])), default=-1) >= 0.5
            ]
            worst = sorted((i for i in range(len(aps)) if not np.isnan(aps[i])), key=lambda i: aps[i])[:5]
            fam_info[fam] = {
                "config": best[fam]["label"],
                "n_failed": len(fails),
                "n_valid": int(np.sum(~np.isnan(aps))),
                "failed_queries": [qids[i] for i in fails],
                "unique_failures": [qids[i] for i in unique],
                "worst": [{"query": qids[i], "ap": float(aps[i])} for i in worst],
            }
        all_fail = [
            qids[i]
            for i in range(len(qids))
            if per_fam and all((not np.isnan(a[i])) and a[i] < threshold for a in per_fam.values())
        ]
        analysis[ds] = {"families": fam_info, "fail_for_all_families": all_fail}
    notes = [
        f"A query 'fails' when its Medium AP < {threshold:g}; 'unique' = fails here while another "
        "family's best config reaches AP >= 0.5."
    ]
    for ds, a in analysis.items():
        for fam, info in sorted(a["families"].items()):
            notes.append(
                f"{ds} / {fam}: {info['n_failed']}/{info['n_valid']} queries fail; "
                f"{len(info['unique_failures'])} unique to this family"
                + (f" (e.g. {', '.join(info['unique_failures'][:3])})" if info["unique_failures"] else "")
                + "."
            )
        notes.append(f"{ds}: {len(a['fail_for_all_families'])} queries fail for every family.")
    return Table(
        "i",
        "Classical vs CNN vs Transformer (best config per family)",
        [
            "dataset",
            "family",
            "best config",
            "mAP E",
            "mAP M",
            "mAP H",
            "orig. mAP",
            "p95 total ms",
            "extract ms/img",
            "B/vec",
        ],
        rows,
        notes,
        extra={"failure_analysis": analysis},
    )


# ---------------------------------------------------------------------- (j) multi-scale
def ablation_j(records: list[Record]) -> Table:
    recs = in_ablation(records, "j")
    rows: list[list[Any]] = []

    def key(r: Record) -> tuple:
        c = r["config"]
        return (
            r["dataset"],
            r["protocol"],
            r["extractor"],
            r["pooling"],
            stable_hash(c["postprocess"]),
            c["distractors"],
        )

    for _, grp in sorted(group_by(recs, key).items()):
        single = next((r for r in grp if r["config"]["extractor"]["scales"] == [1.0]), None)
        for r in sorted(grp, key=lambda r: len(r["config"]["extractor"]["scales"])):
            e, m, h = emh(r)
            rows.append(
                [
                    r["dataset"],
                    f"{r['extractor']}/{r['pooling']}",
                    ",".join(f"{s:g}" for s in r["config"]["extractor"]["scales"]),
                    e,
                    m,
                    h,
                    _delta(m, emh(single)[1]) if single else None,
                    p95(r, "extract"),
                    r["extract_ms_per_image_db"],
                ]
            )
    return Table(
        "j",
        "Multi-scale extraction",
        [
            "dataset",
            "features",
            "scales",
            "mAP E",
            "mAP M",
            "mAP H",
            "dM vs single",
            "p95 extract ms",
            "db extract ms/img",
        ],
        rows,
    )


def all_tables(records: list[Record], hard_threshold: float, headline_protocol: str) -> list[Table]:
    """Every ablation table in report order."""
    b, b_sweep = ablation_b(records)
    return [
        ablation_i(records, hard_threshold),
        b,
        b_sweep,
        ablation_j(records),
        ablation_g(records),
        ablation_c(records),
        ablation_d(records),
        ablation_a(records),
        ablation_f(records),
        ablation_h(records),
        ablation_e(records, headline_protocol),
    ]
