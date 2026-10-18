"""Dataset parsing (original + revisited formats), subsetting, missing images, download fallback."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import pytest

from irbench.data import DatasetUnavailableError, load_dataset
from irbench.data.base import select_subset
from irbench.data.distractors import load_distractors
from irbench.data.download import ensure_dataset_files
from irbench.data.oxford_paris import load_oxford_paris
from irbench.data.toy import generate_toy_distractors


def test_toy_a_oxford_layout_parses_both_protocols(toy_data_dir: Path) -> None:
    ds = load_dataset("toy_a", toy_data_dir, auto_download=False)
    assert set(ds.protocols) == {"original", "revisited"}
    orig, rev = ds.protocols["original"], ds.protocols["revisited"]
    # original database contains the query images (like Oxford5k), revisited does not
    assert len(orig.db_ids) == len(rev.db_ids) + len(rev.queries)
    assert len(orig.queries) == len(rev.queries) == 8
    # 'oxc1_' prefix was stripped and the query exists in the pool
    assert all(q.id in ds.paths and not q.id.startswith("oxc1_") for q in orig.queries)
    assert all(q.bbox is not None and q.bbox[2] > q.bbox[0] for q in rev.queries)
    # good+ok == easy+hard, junk == junk
    for q_o, g_o in zip(orig.queries, orig.gt, strict=True):
        qi = [q.id for q in rev.queries].index(q_o.id)
        rev_pos = {rev.db_ids[i] for i in rev.positives(qi, "medium")}
        assert {orig.db_ids[i] for i in g_o["pos"]} == rev_pos
    assert ds.heldout == "toy_b"


def test_toy_b_nested_paris_layout(toy_data_dir: Path) -> None:
    assert any(p.is_dir() for p in (toy_data_dir / "toy_b" / "jpg").iterdir())
    ds = load_dataset("toy_b", toy_data_dir, auto_download=False)
    assert len(ds.protocols["revisited"].db_ids) > 0
    assert ds.heldout == "toy_a"


def test_missing_and_corrupt_images_are_skipped_and_reported(toy_data_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "toy_a"
    shutil.copytree(toy_data_dir / "toy_a", root)
    gnd = toy_data_dir / "revisitop" / "gnd_rtoy_a.pkl"
    with open(gnd, "rb") as fh:
        cfg = pickle.load(fh)
    easy_row = cfg["gnd"][0]["easy"][0]
    removed = cfg["imlist"][easy_row]
    corrupt = cfg["imlist"][cfg["gnd"][0]["easy"][1]]
    (root / "jpg" / f"{removed}.jpg").unlink()
    (root / "jpg" / f"{corrupt}.jpg").write_bytes(b"not a jpeg")
    (root / ".verify.json").unlink(missing_ok=True)
    ds = load_oxford_paris("toy_a", root, gnd, "toy_b")
    assert removed in ds.missing and corrupt in ds.missing
    assert ds.stats["n_corrupt"] == 1
    rev = ds.protocols["revisited"]
    assert removed not in rev.db_ids and corrupt not in rev.db_ids
    assert ds.stats["revisited_db_dropped"] == 2
    # ground truth was remapped to the shrunken database
    for g in rev.gt:
        for rows in g.values():
            assert all(0 <= r < len(rev.db_ids) for r in rows)


def test_subset_keeps_relevant_images_and_is_seeded(toy_data_dir: Path) -> None:
    ds = load_dataset("toy_a", toy_data_dir, auto_download=False)
    a = select_subset(ds, 30, seed=7)
    b = select_subset(ds, 30, seed=7)
    rev = a.protocols["revisited"]
    assert rev.db_ids == b.protocols["revisited"].db_ids
    assert len(rev.db_ids) <= 30 and len(rev.queries) >= 1
    assert any(rev.positives(i, "medium") for i in range(len(rev.queries)))
    c = select_subset(ds, 30, seed=8)
    assert c.protocols["revisited"].db_ids != rev.db_ids or c.protocols["revisited"].queries != rev.queries


def test_download_failure_names_page_and_folder(tmp_path: Path) -> None:
    with pytest.raises(DatasetUnavailableError) as exc:
        ensure_dataset_files("oxford5k", tmp_path, auto_download=False)
    msg = str(exc.value)
    assert "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/" in msg
    assert "./data/oxford5k" in msg
    assert "./data/revisitop" in msg and "http://cmp.felk.cvut.cz/revisitop/" in msg


def test_manually_placed_files_are_used(tmp_path: Path, toy_data_dir: Path) -> None:
    """Images/gt placed by hand (any layout) count as present; nothing is downloaded."""
    folder = tmp_path / "oxford5k"
    folder.mkdir()
    (folder / "gt").mkdir()
    (folder / "gt" / "all_souls_1_query.txt").write_text("oxc1_x 0 0 1 1\n")
    (tmp_path / "revisitop").mkdir()
    (tmp_path / "revisitop" / "gnd_roxford5k.pkl").write_bytes(b"x")
    with pytest.raises(DatasetUnavailableError) as exc:
        ensure_dataset_files("oxford5k", tmp_path, auto_download=False)
    # only the images are reported missing: gt and gnd were found locally
    assert "oxbuild_images-v1.tgz" in str(exc.value)
    assert "gt_files_170407.tgz" not in str(exc.value)
    assert "gnd_roxford5k.pkl" not in str(exc.value)


def test_distractor_sampling_is_seeded_and_namespaced(tmp_path: Path) -> None:
    folder = generate_toy_distractors(tmp_path / "d", 12, seed=0)
    (folder / "broken.jpg").write_bytes(b"nope")
    a = load_distractors(folder, 8, seed=1)
    b = load_distractors(folder, 8, seed=1)
    assert [i.id for i in a.items] == [i.id for i in b.items]
    assert len(a.items) == 8
    full = load_distractors(folder, 100, seed=1)
    assert len(full.items) == 12 and full.skipped_corrupt == 1 and full.notes
