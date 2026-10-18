"""Revisited protocol (Easy / Medium / Hard) on a toy case with a known answer."""

from __future__ import annotations

import pytest

from irbench.data.base import Protocol, Query
from irbench.eval.metrics import evaluate_protocol


@pytest.fixture
def proto() -> Protocol:
    # rows: 0,1 easy; 2 hard; 3 junk; 4,5 negatives
    return Protocol(
        "revisited",
        db_ids=[f"img{i}" for i in range(6)],
        queries=[Query("q", "", (0, 0, 10, 10))],
        gt=[{"easy": [0, 1], "hard": [2], "junk": [3]}],
    )


def test_setup_definitions(proto: Protocol) -> None:
    assert proto.positives(0, "easy") == [0, 1]
    assert sorted(proto.junk(0, "easy")) == [2, 3]  # hard images are ignored in Easy
    assert proto.positives(0, "medium") == [0, 1, 2]
    assert proto.junk(0, "medium") == [3]
    assert proto.positives(0, "hard") == [2]
    assert sorted(proto.junk(0, "hard")) == [0, 1, 3]  # easy images are ignored in Hard


def test_easy_medium_hard_known_values(proto: Protocol) -> None:
    # ranking: hard(2), easy(0), junk(3), negative(4), easy(1), negative(5)
    res = evaluate_protocol([[2, 0, 3, 4, 1, 5]], proto)
    # Easy: list without {2,3} -> [0, 4, 1, 5]; positives at ranks 0 and 2
    easy = 0.5 + (0.5 + 2 / 3) / 2 * 0.5
    # Medium: list without {3} -> [2, 0, 4, 1, 5]; positives at ranks 0, 1, 3
    medium = (1 / 3) * ((1 + 1) / 2 + (1 + 1) / 2 + (2 / 3 + 3 / 4) / 2)
    # Hard: list without {0,1,3} -> [2, 4, 5]; the single positive at rank 0
    hard = 1.0
    assert res["easy"]["map"] == pytest.approx(100 * easy)
    assert res["medium"]["map"] == pytest.approx(100 * medium)
    assert res["hard"]["map"] == pytest.approx(100 * hard)
    assert res["hard"]["recall@1"] == pytest.approx(100.0)
    assert res["easy"]["recall@1"] == pytest.approx(100.0)  # hard image removed from Easy list


def test_query_without_hard_positives_is_skipped_in_hard(proto: Protocol) -> None:
    proto.gt.append({"easy": [4], "hard": [], "junk": []})
    proto.queries.append(Query("q2", "", None))
    res = evaluate_protocol([[2, 0, 1, 3, 4, 5], [4, 0, 1, 2, 3, 5]], proto)
    assert res["hard"]["n_valid_queries"] == 1
    assert res["easy"]["n_valid_queries"] == 2
