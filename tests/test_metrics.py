"""AP / mAP / Recall@K on tiny synthetic cases with hand-computed answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from irbench.data.base import Protocol, Query
from irbench.eval.metrics import (
    ann_recall,
    average_precision,
    compute_ap,
    evaluate_protocol,
    hit_at_k,
    junk_free_positive_ranks,
)

# Ranking [P, N, P] with 2 positives, trapezoidal rule:
#   j=0, r=0: (1 + 1)/2 * 1/2               = 0.5
#   j=1, r=2: (1/2 + 2/3)/2 * 1/2           = 0.291666...
EXPECTED_PNP = 0.5 + (0.5 + 2 / 3) / 2 * 0.5


def test_compute_ap_known_value() -> None:
    assert compute_ap([0, 2], 2) == pytest.approx(EXPECTED_PNP)


def test_perfect_ranking_has_ap_one() -> None:
    assert average_precision([3, 1, 2, 0], positives=[3, 1], junk=[]) == pytest.approx(1.0)


def test_junk_is_removed_before_scoring() -> None:
    # junk (9) at rank 0 and between the positives must not change AP
    assert average_precision([9, 1, 5, 2], positives=[1, 2], junk=[9]) == pytest.approx(EXPECTED_PNP)
    assert junk_free_positive_ranks([9, 1, 5, 2], [1, 2], [9]) == [0, 2]


def test_junk_counted_as_negative_would_lower_ap() -> None:
    with_junk_ignored = average_precision([9, 1, 2], positives=[1, 2], junk=[9])
    junk_as_negative = average_precision([9, 1, 2], positives=[1, 2], junk=[])
    assert with_junk_ignored == pytest.approx(1.0)
    assert junk_as_negative < with_junk_ignored


def test_unretrieved_positive_is_penalised() -> None:
    # a truncated list that misses one of two positives: only half the recall mass is earned
    assert average_precision([4, 7], positives=[4, 8], junk=[]) == pytest.approx(0.5)


def test_no_positives_is_nan_and_excluded_from_mean() -> None:
    assert math.isnan(average_precision([1, 2], positives=[], junk=[]))
    proto = Protocol(
        "original",
        db_ids=["a", "b", "c"],
        queries=[Query("q1", "", None), Query("q2", "", None)],
        gt=[{"pos": [0], "junk": []}, {"pos": [], "junk": [1]}],
    )
    res = evaluate_protocol([[0, 1, 2], [2, 1, 0]], proto)
    assert res["all"]["map"] == pytest.approx(100.0)
    assert res["all"]["n_valid_queries"] == 1
    assert res["all"]["ap"][1] is None


def test_recall_at_k() -> None:
    assert hit_at_k([5, 6, 1], positives=[1], junk=[], k=1) == 0.0
    assert hit_at_k([5, 6, 1], positives=[1], junk=[], k=3) == 1.0
    # junk before the positive does not consume a slot
    assert hit_at_k([9, 1], positives=[1], junk=[9], k=1) == 1.0


def test_negative_indices_are_ignored() -> None:
    assert average_precision([-1, 0, -1], positives=[0], junk=[]) == pytest.approx(1.0)


def test_ann_recall() -> None:
    exact = np.array([[1, 2, 3], [4, 5, 6]])
    approx = np.array([[1, 3, 9], [-1, 5, 4]])
    assert ann_recall(approx, exact, 3) == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert ann_recall(approx, exact, 1) == pytest.approx(0.5)
