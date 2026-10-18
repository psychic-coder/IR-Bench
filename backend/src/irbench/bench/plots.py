"""PNG plots for every ablation (matplotlib, Agg backend). Inputs are harness records only."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from irbench.bench.ablations import (
    Table,
    ann_k,
    emh,
    feature_label,
    group_by,
    in_ablation,
    p95,
    pareto_front,
    space_key,
)
from irbench.utils import ensure_dir, stable_hash

Record = dict[str, Any]
FAMILY_COLORS = {"classical": "#7f7f7f", "cnn": "#1f77b4", "transformer": "#d62728"}
DATASET_MARKERS = ["o", "s", "^", "D", "v"]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> str:
    ensure_dir(out_dir)
    fig.tight_layout()
    fig.savefig(out_dir / name)
    plt.close(fig)
    return f"plots/{name}"


def _empty(ax: plt.Axes, msg: str = "no data for this ablation") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_a(records: list[Record], out: Path) -> str:
    recs = in_ablation(records, "a")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if not recs:
        _empty(ax)
    for di, (ds, grp) in enumerate(sorted(group_by(recs, lambda r: r["dataset"]).items())):
        for _, sgrp in group_by(grp, space_key).items():
            pts = sorted(
                ((r["memory"]["encoding_bytes_per_vector"], r["headline_map"], r["index"]) for r in sgrp),
                key=lambda t: t[0],
            )
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=DATASET_MARKERS[di % 5], label=f"{ds}: {feature_label(sgrp[0])}")
            for x, y, kind in pts:
                if kind == "flat":
                    ax.scatter([x], [y], marker="*", s=160, color="black", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("encoding bytes per vector (log)")
    ax.set_ylabel("mAP (headline)")
    ax.set_title("IVF-PQ compression vs mAP (star = exact flat)")
    if recs:
        ax.legend(fontsize=7)
    return _save(fig, out, "a_ivfpq_memory.png")


def plot_b(records: list[Record], out: Path) -> str:
    recs = in_ablation(records, "b")
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = sorted(group_by(recs, lambda r: (r["dataset"], r["extractor"])).items())
    pools = ["gap", "gem", "rmac", "patch_gap", "patch_gem", "cls"]
    if not groups:
        _empty(ax)
    width = 0.8 / len(pools)
    for gi, (_key, grp) in enumerate(groups):
        for pi, pool in enumerate(pools):
            vals = [
                r["headline_map"]
                for r in grp
                if r["pooling"] == pool and (r["config"]["pooling"]["p"] == 3.0 or "gem" not in pool)
            ]
            if vals:
                ax.bar(gi + pi * width - 0.4 + width / 2, max(vals), width, color=f"C{pi}", label=pool)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{ds}\n{ext}" for (ds, ext), _ in groups], fontsize=7)
    ax.set_ylabel("mAP (headline)")
    ax.set_title("Pooling per backbone")
    if groups:
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles, strict=False))
        ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    return _save(fig, out, "b_pooling.png")


def plot_b_sweep(records: list[Record], out: Path) -> str:
    recs = [r for r in in_ablation(records, "b") if r["pooling"] in {"gem", "patch_gem"}]
    fig, ax = plt.subplots(figsize=(6, 4))
    if not recs:
        _empty(ax)
    for (ds, ext, pool), grp in sorted(
        group_by(recs, lambda r: (r["dataset"], r["extractor"], r["pooling"])).items()
    ):
        grp = sorted(grp, key=lambda r: r["config"]["pooling"]["p"])
        ax.plot(
            [r["config"]["pooling"]["p"] for r in grp],
            [r["headline_map"] for r in grp],
            marker="o",
            label=f"{ds} {ext}/{pool}",
        )
    ax.set_xlabel("GeM exponent p (p=1 is average pooling)")
    ax.set_ylabel("mAP (headline)")
    ax.set_title("GeM p sweep")
    if recs:
        ax.legend(fontsize=7)
    return _save(fig, out, "b_gem_p_sweep.png")


def _plot_sweep(
    records: list[Record], out: Path, letter: str, kind: str, param: str, fname: str, title: str
) -> str:
    recs = [r for r in in_ablation(records, letter) if r["index"] == kind]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax2 = ax.twinx()
    if not recs:
        _empty(ax)

    def bkey(r: Record) -> tuple:
        bp = {k: v for k, v in r["config"]["index"]["params"].items() if k != param}
        return (space_key(r), stable_hash(bp))

    for ci, (_, grp) in enumerate(sorted(group_by(recs, bkey).items())):
        grp = sorted(grp, key=lambda r: r["index_params_effective"].get(param, 0))
        xs = [r["index_params_effective"].get(param) for r in grp]
        lab = f"{grp[0]['dataset']} {feature_label(grp[0])}"
        ax.plot(xs, [ann_k(r) for r in grp], marker="o", color=f"C{ci}", label=f"{lab} recall")
        ax2.plot(
            xs, [p95(r, "ann_search") for r in grp], marker="x", linestyle="--", color=f"C{ci}", alpha=0.6
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(param)
    ax.set_ylabel("neighbour recall@k vs exact (solid)")
    ax2.set_ylabel("p95 top-k search ms (dashed)")
    ax2.grid(False)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    if recs:
        ax.legend(fontsize=7, loc="lower right")
    return _save(fig, out, fname)


def plot_c(records: list[Record], out: Path) -> str:
    return _plot_sweep(
        records, out, "c", "hnsw", "ef", "c_hnsw_recall_vs_ef.png", "HNSW (Qdrant): recall vs ef"
    )


def plot_d(records: list[Record], out: Path) -> str:
    return _plot_sweep(
        records, out, "d", "ivfpq", "nprobe", "d_ivfpq_recall_vs_nprobe.png", "IVF-PQ: recall vs nprobe"
    )


def plot_e(records: list[Record], out: Path, protocol: str = "revisited") -> str:
    recs = [r for r in records if r.get("status") == "ok" and r["protocol"] == protocol]
    datasets = sorted({r["dataset"] for r in recs})
    fig, axes = plt.subplots(
        1, max(1, len(datasets)), figsize=(6 * max(1, len(datasets)), 4.2), squeeze=False
    )
    if not recs:
        _empty(axes[0][0])
    for ax, ds in zip(axes[0], datasets, strict=False):
        grp = [
            r
            for r in recs
            if r["dataset"] == ds and p95(r, "total") is not None and r["headline_map"] is not None
        ]
        for fam in sorted({r.get("family") for r in grp}):
            pts = [r for r in grp if r.get("family") == fam]
            ax.scatter(
                [p95(r, "total") for r in pts],
                [r["headline_map"] for r in pts],
                s=18,
                alpha=0.7,
                color=FAMILY_COLORS.get(fam or "", "C4"),
                label=fam,
            )
        front = pareto_front([(p95(r, "total"), r["headline_map"], r) for r in grp])
        ax.plot(
            [f[0] for f in front],
            [f[1] for f in front],
            color="black",
            linewidth=1.2,
            drawstyle="steps-post",
            label="Pareto",
        )
        ax.set_xscale("log")
        ax.set_xlabel("p95 end-to-end latency, ms (log)")
        ax.set_ylabel("mAP (headline)")
        ax.set_title(f"{ds}: mAP vs p95 latency")
        ax.legend(fontsize=7)
    return _save(fig, out, "e_pareto_map_vs_p95.png")


def plot_f(records: list[Record], out: Path) -> str:
    recs = in_ablation(records, "f")
    fig, ax = plt.subplots(figsize=(7, 4))
    if not recs:
        _empty(ax)
    groups = sorted(group_by(recs, lambda r: (r["dataset"], feature_label(r), r["index"])).items())
    names = sorted({r["rerank"] for r in recs}, key=lambda n: (n != "none", n))
    width = 0.8 / max(1, len(names))
    for gi, (_, grp) in enumerate(groups):
        for ni, name in enumerate(names):
            r = next((x for x in grp if x["rerank"] == name), None)
            if r is None:
                continue
            x = gi + ni * width - 0.4 + width / 2
            ax.bar(x, emh(r)[1], width, color=f"C{ni}", label=name if gi == 0 else None)
            lat = p95(r, "rerank")
            if lat is not None:
                ax.text(x, (emh(r)[1] or 0) + 0.5, f"{lat:.1f}ms", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k, _ in groups], fontsize=7)
    ax.set_ylabel("mAP Medium (labels: p95 rerank ms)")
    ax.set_title("Re-ranking effect and latency cost")
    if recs:
        ax.legend(fontsize=7)
    return _save(fig, out, "f_reranking.png")


def plot_g(records: list[Record], out: Path) -> str:
    recs = in_ablation(records, "g")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if not recs:
        _empty(ax)
    for ci, ((ds, feat), grp) in enumerate(
        sorted(group_by(recs, lambda r: (r["dataset"], f"{r['extractor']}/{r['pooling']}")).items())
    ):
        for mode, ls in (("pca", ":"), ("whiten", "-")):
            pts = sorted(
                (r["vector_dim"], emh(r)[1]) for r in grp if r["config"]["postprocess"]["pca"] == mode
            )
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    linestyle=ls,
                    color=f"C{ci}",
                    label=f"{ds} {feat} {mode}",
                )
        base = next((r for r in grp if r["config"]["postprocess"]["pca"] == "none"), None)
        if base is not None:
            ax.axhline(emh(base)[1], color=f"C{ci}", linestyle="--", alpha=0.5, label=f"{ds} {feat} no PCA")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("output dimension")
    ax.set_ylabel("mAP Medium")
    ax.set_title("PCA / whitening (fit on held-out dataset)")
    if recs:
        ax.legend(fontsize=6)
    return _save(fig, out, "g_pca_whitening.png")


def plot_h(records: list[Record], out: Path) -> str:
    recs = in_ablation(records, "h")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if not recs:
        _empty(ax)
    for ci, (_, grp) in enumerate(sorted(group_by(recs, lambda r: (space_key(r, True), r["index"])).items())):
        grp = sorted(grp, key=lambda r: r["n_distractors"])
        ax.plot(
            [r["n_distractors"] for r in grp],
            [emh(r)[1] for r in grp],
            marker="o",
            color=f"C{ci}",
            label=f"{grp[0]['dataset']} {feature_label(grp[0])} {grp[0]['index']}",
        )
    ax.set_xlabel("number of distractor images")
    ax.set_ylabel("mAP Medium")
    ax.set_title("Scale test")
    if recs:
        ax.legend(fontsize=7)
    return _save(fig, out, "h_scale_distractors.png")


def plot_i(records: list[Record], out: Path, table: Table) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = table.rows
    if not rows:
        _empty(ax)
    labels = [f"{r[0]}\n{r[1]}" for r in rows]
    x = np.arange(len(rows))
    for si, (name, col) in enumerate((("Easy", 3), ("Medium", 4), ("Hard", 5))):
        vals = [np.nan if r[col] is None else r[col] for r in rows]
        ax.bar(x + (si - 1) * 0.27, vals, 0.27, label=name, color=f"C{si}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mAP (revisited)")
    ax.set_title("Best configuration per family")
    if rows:
        ax.legend(fontsize=7)
    return _save(fig, out, "i_families.png")


def plot_j(records: list[Record], out: Path) -> str:
    recs = in_ablation(records, "j")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if not recs:
        _empty(ax)
    groups = sorted(group_by(recs, lambda r: (r["dataset"], f"{r['extractor']}/{r['pooling']}")).items())
    for gi, (_, grp) in enumerate(groups):
        grp = sorted(grp, key=lambda r: len(r["config"]["extractor"]["scales"]))
        for si, r in enumerate(grp):
            ax.bar(
                gi + si * 0.35 - 0.175,
                emh(r)[1],
                0.35,
                color=f"C{si}",
                label=("single-scale" if si == 0 else "multi-scale") if gi == 0 else None,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k, _ in groups], fontsize=7)
    ax.set_ylabel("mAP Medium")
    ax.set_title("Multi-scale extraction")
    if recs:
        ax.legend(fontsize=7)
    return _save(fig, out, "j_multiscale.png")


PLOTTERS: dict[str, Callable[..., str]] = {
    "a": plot_a,
    "b": plot_b,
    "b_sweep": plot_b_sweep,
    "c": plot_c,
    "d": plot_d,
    "f": plot_f,
    "g": plot_g,
    "h": plot_h,
    "j": plot_j,
}


def make_plots(records: list[Record], tables: list[Table], out_dir: Path, headline_protocol: str) -> None:
    """Render one PNG per table and store its relative path on the table."""
    for t in tables:
        if t.key == "e":
            t.plot = plot_e(records, out_dir, headline_protocol)
        elif t.key == "i":
            t.plot = plot_i(records, out_dir, t)
        elif t.key in PLOTTERS:
            t.plot = PLOTTERS[t.key](records, out_dir)
