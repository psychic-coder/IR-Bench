"""Procedurally generated 'toy landmark' datasets for tests, CI and offline smoke runs.

Each toy dataset is written to disk in *exactly* the Oxford/Paris formats so the real
parsers, evaluators and harness are exercised end-to-end:

* ``toy_a`` mimics Oxford: flat ``jpg/`` folder, query names prefixed with ``oxc1_``.
* ``toy_b`` mimics Paris: images nested in ``jpg/<landmark>/`` folders.
* Both get original-protocol ``gt/*.txt`` files and a revisited ``gnd_r<name>.pkl``.

A landmark is a random textured scene; database images are random affine views of it.
Easy views are mild, hard views are strong (scale/rotation/occlusion/photometric), junk
views show only a small part of the scene. Negatives are views of unrelated scenes.
Numbers produced on toy data say nothing about real-world retrieval quality.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from irbench.utils import ensure_dir

TOY_DATASETS = ("toy_a", "toy_b")
OUT_W, OUT_H = 512, 384
SCENE_W, SCENE_H = 800, 600


@dataclass(frozen=True)
class ToySpec:
    landmarks: int = 6
    queries_per_landmark: int = 2
    easy_per_landmark: int = 6
    hard_per_landmark: int = 4
    junk_per_landmark: int = 2
    negatives: int = 30


def _scene(rng: np.random.Generator) -> np.ndarray:
    """A random textured scene with many corners (ORB-friendly)."""
    base = rng.integers(40, 200, size=3)
    img = np.empty((SCENE_H, SCENE_W, 3), dtype=np.uint8)
    grad = np.linspace(0, 1, SCENE_W)[None, :, None]
    img[:] = np.clip(base[None, None, :] * (0.6 + 0.6 * grad), 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(35, 60))):
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        kind = int(rng.integers(0, 4))
        x, y = int(rng.integers(0, SCENE_W)), int(rng.integers(0, SCENE_H))
        if kind == 0:
            w, h = int(rng.integers(20, 160)), int(rng.integers(20, 160))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        elif kind == 1:
            cv2.circle(img, (x, y), int(rng.integers(10, 80)), color, -1)
        elif kind == 2:
            pts = rng.integers(0, [SCENE_W, SCENE_H], size=(int(rng.integers(3, 7)), 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], color)
        else:
            x2, y2 = int(rng.integers(0, SCENE_W)), int(rng.integers(0, SCENE_H))
            cv2.line(img, (x, y), (x2, y2), color, int(rng.integers(2, 8)))
    checker = (np.indices((SCENE_H // 8, SCENE_W // 8)).sum(axis=0) % 2).astype(np.uint8)
    checker = cv2.resize(checker, (SCENE_W, SCENE_H), interpolation=cv2.INTER_NEAREST)
    x0, y0 = int(rng.integers(0, SCENE_W - 200)), int(rng.integers(0, SCENE_H - 150))
    img[y0 : y0 + 150, x0 : x0 + 200][checker[y0 : y0 + 150, x0 : x0 + 200] == 1] //= 2
    return img


def _background(rng: np.random.Generator) -> np.ndarray:
    noise = rng.integers(0, 255, size=(OUT_H // 16, OUT_W // 16, 3), dtype=np.uint8)
    return cv2.resize(noise, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)


def _view(scene: np.ndarray, rng: np.random.Generator, level: str) -> tuple[np.ndarray, tuple[float, ...]]:
    """Render a view of ``scene``; returns the image and the bbox of the scene centre."""
    if level == "easy":
        angle, scale, shift, occl = rng.uniform(-8, 8), rng.uniform(0.85, 1.1), 30, 0.0
    elif level == "hard":
        angle = rng.uniform(-25, 25)
        scale = rng.choice([rng.uniform(0.45, 0.6), rng.uniform(1.4, 1.7)])
        shift, occl = 80, rng.uniform(0.2, 0.35)
    elif level == "junk":
        angle, scale, shift, occl = rng.uniform(-10, 10), rng.uniform(2.6, 3.2), 120, 0.0
    else:  # query
        angle, scale, shift, occl = rng.uniform(-4, 4), rng.uniform(0.75, 0.85), 15, 0.0
    fit = min(OUT_W / SCENE_W, OUT_H / SCENE_H)
    M = cv2.getRotationMatrix2D((SCENE_W / 2, SCENE_H / 2), float(angle), float(scale * fit))
    M[0, 2] += OUT_W / 2 - SCENE_W / 2 + rng.uniform(-shift, shift)
    M[1, 2] += OUT_H / 2 - SCENE_H / 2 + rng.uniform(-shift, shift)
    warped = cv2.warpAffine(scene, M, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(
        np.full(scene.shape[:2], 255, np.uint8), M, (OUT_W, OUT_H), flags=cv2.INTER_NEAREST
    )
    out = _background(rng)
    out[valid > 0] = warped[valid > 0]
    if occl > 0:
        ow, oh = int(OUT_W * np.sqrt(occl)), int(OUT_H * np.sqrt(occl))
        ox, oy = int(rng.integers(0, OUT_W - ow)), int(rng.integers(0, OUT_H - oh))
        out[oy : oy + oh, ox : ox + ow] = rng.integers(0, 255, size=3)
    if level == "hard":
        alpha, beta = rng.uniform(0.6, 1.4), rng.uniform(-40, 40)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    corners = np.array(
        [
            [SCENE_W * 0.15, SCENE_H * 0.15],
            [SCENE_W * 0.85, SCENE_H * 0.15],
            [SCENE_W * 0.85, SCENE_H * 0.85],
            [SCENE_W * 0.15, SCENE_H * 0.85],
        ]
    )
    proj = corners @ M[:, :2].T + M[:, 2]
    x1, y1 = np.clip(proj.min(axis=0), 0, [OUT_W - 2, OUT_H - 2])
    x2, y2 = np.clip(proj.max(axis=0), [x1 + 1, y1 + 1], [OUT_W, OUT_H])
    return out, (float(x1), float(y1), float(x2), float(y2))


def _save(img: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def generate_toy_dataset(name: str, data_dir: Path, seed: int = 0, spec: ToySpec | None = None) -> Path:
    """Write toy dataset ``name`` under ``data_dir/name`` and return its root.

    Idempotent: an existing dataset is reused when ``spec`` is ``None`` or matches the one it
    was generated with; otherwise the folder is wiped and regenerated (never mixed).
    """
    if name not in TOY_DATASETS:
        raise KeyError(f"unknown toy dataset '{name}'")
    root = data_dir / name
    marker = root / ".generated.json"
    if marker.exists():
        try:
            existing = json.loads(marker.read_text())
        except ValueError:
            existing = None
        if existing is not None and (spec is None or existing == _fingerprint(seed, spec)):
            return root
    spec = spec or ToySpec()
    fingerprint = _fingerprint(seed, spec)
    if root.exists():
        shutil.rmtree(root)
    oxford_style = name == "toy_a"
    rng = np.random.default_rng(seed + (0 if oxford_style else 10_000))
    jpg_root = ensure_dir(root / "jpg")
    gt_dir = ensure_dir(root / "gt")
    names = ["tower", "bridge", "church", "museum", "palace", "gate", "arch", "hall"]

    def img_path(landmark: str, stem: str) -> Path:
        return jpg_root / stem if oxford_style else jpg_root / landmark / stem

    imlist: list[str] = []
    qimlist: list[str] = []
    gnd: list[dict[str, object]] = []
    counter = 0

    def next_id(landmark: str) -> str:
        nonlocal counter
        counter += 1
        return f"{name}_{landmark}_{counter:06d}"

    for li in range(spec.landmarks):
        landmark = names[li % len(names)] + (str(li // len(names)) if li >= len(names) else "")
        scene = _scene(rng)
        groups: dict[str, list[str]] = {"easy": [], "hard": [], "junk": []}
        for level, count in (
            ("easy", spec.easy_per_landmark),
            ("hard", spec.hard_per_landmark),
            ("junk", spec.junk_per_landmark),
        ):
            for _ in range(count):
                iid = next_id(landmark)
                img, _ = _view(scene, rng, level)
                _save(img, img_path(landmark, f"{iid}.jpg"))
                groups[level].append(iid)
                imlist.append(iid)
        for qn in range(1, spec.queries_per_landmark + 1):
            qid = next_id(landmark)
            img, bbox = _view(scene, rng, "query")
            _save(img, img_path(landmark, f"{qid}.jpg"))
            qimlist.append(qid)
            gnd.append(
                {"bbx": list(bbox), "easy": groups["easy"], "hard": groups["hard"], "junk": groups["junk"]}
            )
            prefix = "oxc1_" if oxford_style else ""
            stem = f"{landmark}_{qn}"
            (gt_dir / f"{stem}_query.txt").write_text(
                f"{prefix}{qid} {bbox[0]:.1f} {bbox[1]:.1f} {bbox[2]:.1f} {bbox[3]:.1f}\n"
            )
            (gt_dir / f"{stem}_good.txt").write_text("".join(f"{i}\n" for i in groups["easy"]))
            (gt_dir / f"{stem}_ok.txt").write_text("".join(f"{i}\n" for i in groups["hard"]))
            (gt_dir / f"{stem}_junk.txt").write_text("".join(f"{i}\n" for i in groups["junk"]))

    for _ in range(spec.negatives):
        iid = next_id("negative")
        img, _ = _view(_scene(rng), rng, "easy")
        _save(img, img_path("negative", f"{iid}.jpg"))
        imlist.append(iid)

    index = {iid: r for r, iid in enumerate(imlist)}
    gnd_rows = [
        {
            "bbx": g["bbx"],
            "easy": [index[i] for i in g["easy"]],  # type: ignore[union-attr]
            "hard": [index[i] for i in g["hard"]],  # type: ignore[union-attr]
            "junk": [index[i] for i in g["junk"]],  # type: ignore[union-attr]
        }
        for g in gnd
    ]
    gnd_dir = ensure_dir(data_dir / "revisitop")
    with open(gnd_dir / f"gnd_r{name}.pkl", "wb") as fh:
        pickle.dump({"imlist": imlist, "qimlist": qimlist, "gnd": gnd_rows}, fh)
    marker.write_text(json.dumps(fingerprint))
    return root


def _fingerprint(seed: int, spec: ToySpec) -> dict:
    return {"seed": seed, "spec": dict(spec.__dict__), "v": 1}


def generate_toy_distractors(folder: Path, count: int, seed: int = 0) -> Path:
    """Write ``count`` unrelated scene views into ``folder`` (idempotent)."""
    ensure_dir(folder)
    existing = sorted(folder.glob("toy_distractor_*.jpg"))
    if len(existing) >= count:
        return folder
    rng = np.random.default_rng(seed + 99_999)
    for i in range(count):
        img, _ = _view(_scene(rng), rng, "easy")
        path = folder / f"toy_distractor_{i:06d}.jpg"
        if not path.exists():
            _save(img, path)
    return folder
