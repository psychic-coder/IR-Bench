"""Retrieval metrics: official AP (original & revisited protocols), mAP, Recall@K, ANN recall.

AP follows the reference implementations (Oxford ``compute_ap.cpp`` and revisitop
``compute_ap``): junk images are removed from the ranked list, then precision is
integrated with the trapezoidal rule over recall steps of ``1 / n_positives``::

    AP = sum_j  (P_{j-1} + P_j) / 2 * (1 / n_pos),
    P_{j-1} = j / r_j  (1 if r_j == 0),   P_j = (j + 1) / (r_j + 1)

where ``r_j`` is the 0-based junk-free rank of the j-th retrieved positive. Positives that
do not appear in the (possibly truncated) ranked list contribute nothing, so an index that
fails to return a positive is penalised. Queries without positives score NaN and are
excluded from the mean, as in the official code.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from irbench.data.base import Protocol

REVISITED_SETUPS = ("easy", "medium", "hard")
RECALL_KS = (1, 5, 10)


def compute_ap(ranks: Sequence[int], n_positives: int) -> float:
    """Trapezoidal AP from sorted 0-based junk-free ranks of the retrieved positives."""
    if n_positives <= 0:
        return float("nan")
    ap = 0.0
    recall_step = 1.0 / n_positives
    for j, rank in enumerate(ranks):
        precision_0 = 1.0 if rank == 0 else j / rank
        precision_1 = (j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def junk_free_positive_ranks(
    ranking: Sequence[int], positives: Sequence[int], junk: Sequence[int]
) -> list[int]:
    """0-based ranks of positives in ``ranking`` after deleting junk entries."""
    pos = set(int(p) for p in positives)
    jnk = set(int(j) for j in junk) - pos
    ranks: list[int] = []
    shift = 0
    for r, idx in enumerate(ranking):
        idx = int(idx)
        if idx < 0 or idx in jnk:  # empty ANN slots and junk are both deleted from the list
            shift += 1
            continue
        if idx in pos:
            ranks.append(r - shift)
    return ranks


def average_precision(ranking: Sequence[int], positives: Sequence[int], junk: Sequence[int]) -> float:
    """AP of one ranked list (database row indices, best first)."""
    if len(positives) == 0:
        return float("nan")
    return compute_ap(junk_free_positive_ranks(ranking, positives, junk), len(set(positives)))


def hit_at_k(ranking: Sequence[int], positives: Sequence[int], junk: Sequence[int], k: int) -> float:
    """1.0 if a positive appears in the top-``k`` junk-free results (NaN if no positives)."""
    if len(positives) == 0:
        return float("nan")
    ranks = junk_free_positive_ranks(ranking, positives, junk)
    return 1.0 if ranks and ranks[0] < k else 0.0


def _nanmean(values: Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def evaluate_protocol(
    rankings: Sequence[Sequence[int]], protocol: Protocol, ks: Sequence[int] = RECALL_KS
) -> dict:
    """mAP / per-query AP / Recall@K for every setup of a protocol.

    Returns ``{setup: {"map", "recall@k"..., "ap": [...], "n_valid_queries"}}`` where the
    setups are ``("all",)`` for the original protocol and easy/medium/hard for revisited.
    mAP and recall values are percentages (0-100) to match the literature.
    """
    if len(rankings) != len(protocol.queries):
        raise ValueError("one ranking per query is required")
    setups = ("all",) if protocol.name == "original" else REVISITED_SETUPS
    out: dict[str, dict] = {}
    for setup in setups:
        s = None if setup == "all" else setup
        aps: list[float] = []
        hits: dict[int, list[float]] = {k: [] for k in ks}
        for qi, ranking in enumerate(rankings):
            pos = protocol.positives(qi, s)
            junk = protocol.junk(qi, s)
            aps.append(average_precision(ranking, pos, junk))
            for k in ks:
                hits[k].append(hit_at_k(ranking, pos, junk, k))
        m = _nanmean(aps)
        entry: dict = {
            "map": None if m is None else 100.0 * m,
            "ap": [None if np.isnan(a) else float(a) for a in aps],
            "n_valid_queries": int(np.sum(~np.isnan(np.asarray(aps, dtype=np.float64)))),
        }
        for k in ks:
            r = _nanmean(hits[k])
            entry[f"recall@{k}"] = None if r is None else 100.0 * r
        out[setup] = entry
    return out


def ann_recall(approx: np.ndarray, exact: np.ndarray, k: int) -> float:
    """Mean |approx_top_k ∩ exact_top_k| / k over queries (neighbour recall of an ANN index)."""
    k = min(k, exact.shape[1], approx.shape[1])
    if k == 0:
        return float("nan")
    total = 0.0
    for a, e in zip(approx[:, :k], exact[:, :k], strict=True):
        total += len(set(int(x) for x in a if x >= 0) & set(int(x) for x in e)) / k
    return total / approx.shape[0]
