#!/usr/bin/env python3
"""
generate_toy_a.py - Synthetic image-retrieval dataset generator ("toy_a").

Produces a small, fully synthetic landmark-retrieval benchmark laid out like
the Oxford/Paris (original) and RevisitOP (revisited) datasets:

    <output>/toy_a/jpg/<image_id>.jpg
    <output>/toy_a/gt/<landmark>_<n>_{query,good,ok,junk}.txt
    <output>/revisitop/gnd_rtoy_a.pkl

Requirements: Python 3.10+, Pillow, NumPy. No internet, no external data,
no pretrained models.

Usage:
    python generate_toy_a.py --output data --seed 42
    python generate_toy_a.py --output data --validate-only
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATASET = "toy_a"
WIDTH, HEIGHT = 512, 384                 # final image size
CANVAS_W, CANVAS_H = 768, 576            # canonical scene size (rendered larger, then warped)
LANDMARK_BOX = (96, 60, 672, 516)        # landmark region in canonical coordinates

LANDMARKS = ["tower", "bridge", "church", "museum", "palace", "gate", "arch", "hall"]
N_QUERY = 8
N_EASY = 8
N_HARD = 5
N_JUNK = 3
N_NEGATIVE = 30
NEGATIVE_NAME = "negative"
JPEG_QUALITY = 92

# Category-specific palettes: (background, primary, secondary, accent)
PALETTES = {
    "tower":  ((210, 225, 245), (180, 40, 40),   (250, 200, 60),  (30, 30, 90)),
    "bridge": ((200, 240, 220), (20, 90, 160),   (240, 120, 30),  (60, 60, 60)),
    "church": ((245, 235, 210), (110, 60, 150),  (230, 230, 240), (200, 30, 90)),
    "museum": ((230, 230, 230), (30, 120, 60),   (220, 180, 40),  (120, 20, 20)),
    "palace": ((250, 220, 230), (200, 140, 20),  (90, 30, 120),   (20, 150, 160)),
    "gate":   ((220, 210, 250), (160, 30, 30),   (40, 40, 40),    (250, 250, 120)),
    "arch":   ((250, 240, 200), (60, 110, 180),  (200, 60, 160),  (40, 140, 40)),
    "hall":   ((215, 245, 245), (90, 70, 40),    (230, 90, 60),   (30, 60, 140)),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_font(size: int) -> ImageFont.ImageFont:
    """Pillow's bundled font (no external font files needed)."""
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def jitter(color, rng, amount=12):
    return tuple(int(np.clip(c + rng.integers(-amount, amount + 1), 0, 255)) for c in color)


# --------------------------------------------------------------------------
# Canonical landmark scenes
# --------------------------------------------------------------------------
def draw_texture(draw: ImageDraw.ImageDraw, idx: int, bg, sec):
    """Category-specific background texture: grid, checker or stripes."""
    kind = idx % 3
    period = 24 + 8 * idx
    light = tuple(int(0.8 * b + 0.2 * s) for b, s in zip(bg, sec))
    if kind == 0:  # grid
        for x in range(0, CANVAS_W, period):
            draw.line([(x, 0), (x, CANVAS_H)], fill=light, width=2)
        for y in range(0, CANVAS_H, period):
            draw.line([(0, y), (CANVAS_W, y)], fill=light, width=2)
    elif kind == 1:  # checker
        for yi, y in enumerate(range(0, CANVAS_H, period)):
            for xi, x in enumerate(range(0, CANVAS_W, period)):
                if (xi + yi) % 2 == 0:
                    draw.rectangle([x, y, x + period - 1, y + period - 1], fill=light)
    else:  # diagonal stripes
        for k in range(-CANVAS_H, CANVAS_W, period):
            draw.line([(k, 0), (k + CANVAS_H, CANVAS_H)], fill=light, width=6)


def draw_structure(draw: ImageDraw.ImageDraw, name: str, p, s, a):
    """The main geometric 'building' for each landmark."""
    if name == "tower":
        for i, w in enumerate([220, 170, 120, 80]):
            y0 = 470 - (i + 1) * 90
            draw.rectangle([384 - w // 2, y0, 384 + w // 2, y0 + 90], fill=p if i % 2 == 0 else s, outline=a, width=4)
        draw.polygon([(344, 110), (424, 110), (384, 70)], fill=a)
        draw.ellipse([359, 300, 409, 350], fill=s, outline=a, width=3)
    elif name == "bridge":
        draw.rectangle([110, 300, 658, 330], fill=p)
        for x0 in (140, 300, 460):
            draw.arc([x0, 250, x0 + 170, 420], 180, 360, fill=s, width=14)
            draw.rectangle([x0 + 75, 330, x0 + 95, 470], fill=a)
        for x in range(130, 650, 40):
            draw.line([(x, 300), (x, 260)], fill=a, width=4)
        draw.rectangle([110, 470, 658, 500], fill=(70, 140, 210))
    elif name == "church":
        draw.rectangle([230, 260, 538, 480], fill=p, outline=a, width=4)
        draw.polygon([(210, 260), (558, 260), (384, 140)], fill=s, outline=a)
        draw.rectangle([374, 70, 394, 150], fill=a)
        draw.rectangle([350, 92, 418, 110], fill=a)
        draw.ellipse([344, 190, 424, 250], fill=a)
        draw.rectangle([354, 380, 414, 480], fill=s)
    elif name == "museum":
        draw.polygon([(150, 200), (618, 200), (384, 100)], fill=s, outline=a)
        draw.rectangle([150, 200, 618, 230], fill=a)
        for x in range(170, 600, 70):
            draw.rectangle([x, 230, x + 30, 440], fill=p)
        draw.rectangle([130, 440, 638, 480], fill=a)
        draw.ellipse([364, 130, 404, 170], fill=p)
    elif name == "palace":
        draw.rectangle([140, 250, 628, 480], fill=p, outline=a, width=4)
        for cx in (200, 384, 568):
            draw.pieslice([cx - 70, 130, cx + 70, 290], 180, 360, fill=s)
            draw.rectangle([cx - 4, 90, cx + 4, 140], fill=a)
            draw.ellipse([cx - 12, 78, cx + 12, 102], fill=a)
        for x in range(170, 610, 60):
            draw.rectangle([x, 310, x + 30, 360], fill=s)
            draw.rectangle([x, 400, x + 30, 450], fill=a)
    elif name == "gate":
        draw.rectangle([180, 200, 260, 480], fill=p)
        draw.rectangle([508, 200, 588, 480], fill=p)
        draw.rectangle([140, 140, 628, 180], fill=s)
        draw.rectangle([170, 200, 598, 225], fill=s)
        draw.polygon([(120, 140), (648, 140), (600, 110), (168, 110)], fill=p)
        for x in range(290, 490, 30):
            draw.line([(x, 230), (x, 480)], fill=a, width=6)
    elif name == "arch":
        draw.pieslice([180, 120, 588, 520], 180, 360, fill=p)
        draw.rectangle([180, 320, 588, 480], fill=p)
        draw.pieslice([270, 210, 498, 440], 180, 360, fill=a)
        draw.rectangle([270, 325, 498, 480], fill=a)
        for i in range(7):
            ang = math.pi + i * math.pi / 6
            x = 384 + 180 * math.cos(ang)
            y = 320 + 180 * math.sin(ang)
            draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill=s)
    elif name == "hall":
        draw.rectangle([120, 180, 648, 480], fill=p, outline=a, width=5)
        draw.rectangle([120, 150, 648, 180], fill=s)
        for row, y in enumerate(range(210, 450, 60)):
            for col, x in enumerate(range(150, 620, 60)):
                fill = s if (row + col) % 2 == 0 else a
                draw.rectangle([x, y, x + 36, y + 36], fill=fill)
        draw.ellipse([354, 90, 414, 150], fill=a)


def draw_symbols(draw: ImageDraw.ImageDraw, rng: np.random.Generator, p, s, a):
    """Category-specific 'signature' symbols at fixed (per-category) positions."""
    x0, y0, x1, y1 = LANDMARK_BOX
    for _ in range(6):
        cx = int(rng.integers(x0, x1)); cy = int(rng.integers(y0, y1))
        r = int(rng.integers(10, 26))
        kind = int(rng.integers(0, 4))
        col = [p, s, a][int(rng.integers(0, 3))]
        if kind == 0:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=5)
        elif kind == 1:
            draw.regular_polygon((cx, cy, r), n_sides=int(rng.integers(3, 7)), fill=col)
        elif kind == 2:
            draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=col, width=5)
            draw.line([(cx - r, cy + r), (cx + r, cy - r)], fill=col, width=5)
        else:
            draw.rectangle([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=col)


def render_landmark(idx: int, name: str, seed: int) -> Image.Image:
    bg, p, s, a = PALETTES[name]
    rng = np.random.default_rng([seed, 1000 + idx])
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), bg)
    d = ImageDraw.Draw(img)
    draw_texture(d, idx, bg, s)
    draw_structure(d, name, p, s, a)
    draw_symbols(d, rng, p, s, a)
    # Label banner
    label = f"LANDMARK_{idx + 1:02d}"
    d.rectangle([250, 500, 518, 548], fill=a)
    d.text((384, 524), label, fill=(255, 255, 255), font=get_font(30), anchor="mm")
    d.text((60, 30), name.upper(), fill=a, font=get_font(26))
    return img


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------
@dataclass
class Level:
    rot: float          # max |rotation| degrees
    scale: tuple        # (min, max) zoom factor
    shift: float        # max |translation| px (output space)
    bc: float           # max brightness/contrast deviation
    blur: float         # max gaussian blur radius
    noise: float        # max noise std
    occl: tuple         # (min, max) number of occluders
    occl_size: tuple    # (min, max) occluder side as fraction of image
    crop_prob: float = 0.0          # probability of a zoomed-in crop
    focus: bool = False             # junk: zoom heavily into a small part


LEVELS = {
    "query": Level(rot=4,  scale=(0.95, 1.05), shift=12, bc=0.08, blur=0.6, noise=3,  occl=(0, 0), occl_size=(0, 0)),
    "easy":  Level(rot=8,  scale=(0.90, 1.12), shift=25, bc=0.15, blur=1.0, noise=5,  occl=(0, 1), occl_size=(0.08, 0.15)),
    "hard":  Level(rot=30, scale=(0.60, 1.45), shift=70, bc=0.40, blur=2.5, noise=14, occl=(1, 2), occl_size=(0.15, 0.30), crop_prob=0.5),
    "junk":  Level(rot=45, scale=(2.4, 3.4),   shift=40, bc=0.45, blur=3.0, noise=18, occl=(2, 3), occl_size=(0.25, 0.40), focus=True),
}


def forward_matrix(rot_deg, scale, tx, ty, cx=CANVAS_W / 2, cy=CANVAS_H / 2):
    """Canonical -> output affine (3x3)."""
    s0 = WIDTH / CANVAS_W
    th = math.radians(rot_deg)
    c, sn = math.cos(th), math.sin(th)
    k = s0 * scale
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    RS = np.array([[k * c, -k * sn, 0], [k * sn, k * c, 0], [0, 0, 1]], float)
    T2 = np.array([[1, 0, WIDTH / 2 + tx], [0, 1, HEIGHT / 2 + ty], [0, 0, 1]], float)
    return T2 @ RS @ T1


def bbox_of_region(M, box):
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], float).T
    out = M @ pts
    bx0 = int(np.clip(np.floor(out[0].min()), 0, WIDTH - 2))
    by0 = int(np.clip(np.floor(out[1].min()), 0, HEIGHT - 2))
    bx1 = int(np.clip(np.ceil(out[0].max()), bx0 + 1, WIDTH - 1))
    by1 = int(np.clip(np.ceil(out[1].max()), by0 + 1, HEIGHT - 1))
    return [bx0, by0, bx1, by1]


def make_view(canon: Image.Image, level: Level, rng: np.random.Generator):
    """Apply a random geometric + photometric transformation. Returns (img, bbox)."""
    rot = float(rng.uniform(-level.rot, level.rot))
    scale = float(rng.uniform(*level.scale))
    tx = float(rng.uniform(-level.shift, level.shift))
    ty = float(rng.uniform(-level.shift, level.shift))
    cx, cy = CANVAS_W / 2, CANVAS_H / 2
    if level.focus or rng.random() < level.crop_prob:
        # zoom/crop around a random point of the landmark (partial view)
        x0, y0, x1, y1 = LANDMARK_BOX
        cx = float(rng.uniform(x0 + 40, x1 - 40))
        cy = float(rng.uniform(y0 + 40, y1 - 40))
        if not level.focus:
            scale *= float(rng.uniform(1.3, 1.8))

    M = forward_matrix(rot, scale, tx, ty, cx, cy)
    Minv = np.linalg.inv(M)
    fill = tuple(int(v) for v in rng.integers(0, 256, 3))
    img = canon.transform((WIDTH, HEIGHT), Image.AFFINE, data=tuple(Minv[:2].ravel()),
                          resample=Image.BILINEAR, fillcolor=fill)
    bbox = bbox_of_region(M, LANDMARK_BOX)

    # photometric
    img = ImageEnhance.Brightness(img).enhance(1 + float(rng.uniform(-level.bc, level.bc)))
    img = ImageEnhance.Contrast(img).enhance(1 + float(rng.uniform(-level.bc, level.bc)))
    r = float(rng.uniform(0, level.blur))
    if r > 0.15:
        img = img.filter(ImageFilter.GaussianBlur(r))

    # occlusion
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(level.occl[0], level.occl[1] + 1))):
        w = int(WIDTH * rng.uniform(*level.occl_size))
        h = int(HEIGHT * rng.uniform(*level.occl_size))
        x = int(rng.integers(0, WIDTH - w)); y = int(rng.integers(0, HEIGHT - h))
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            d.rectangle([x, y, x + w, y + h], fill=col)
        else:
            d.ellipse([x, y, x + w, y + h], fill=col)

    # noise
    sd = float(rng.uniform(0, level.noise))
    arr = np.asarray(img).astype(np.float32)
    if sd > 0.5:
        arr = arr + rng.normal(0, sd, arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return img, bbox


def make_negative(rng: np.random.Generator) -> Image.Image:
    """Unrelated abstract image: gradients, blobs, random strokes, no landmark/label."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    c0 = rng.integers(0, 256, 3).astype(np.float32)
    c1 = rng.integers(0, 256, 3).astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    t = (xx * math.cos(ang) + yy * math.sin(ang))
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    arr = c0[None, None] * (1 - t[..., None]) + c1[None, None] * t[..., None]
    freq = rng.uniform(0.01, 0.05)
    arr += 25 * np.sin(freq * xx + rng.uniform(0, 6))[..., None] * np.cos(freq * yy)[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(6, 20))):
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        pts = [(int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))) for _ in range(int(rng.integers(3, 7)))]
        kind = int(rng.integers(0, 3))
        if kind == 0:
            d.line(pts, fill=col, width=int(rng.integers(2, 10)), joint="curve")
        elif kind == 1:
            x, y = pts[0]; r = int(rng.integers(8, 70))
            d.ellipse([x - r, y - int(r * rng.uniform(0.4, 1.6)), x + r, y + r], fill=col)
        else:
            d.polygon(pts, fill=col)
    img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0, 2.0))))
    arr = np.asarray(img).astype(np.float32) + rng.normal(0, rng.uniform(2, 12), (HEIGHT, WIDTH, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
@dataclass
class Plan:
    queries: list = field(default_factory=list)      # (qid, landmark, n, bbox)
    db: list = field(default_factory=list)           # image ids in imlist order
    groups: dict = field(default_factory=dict)       # landmark -> {"easy": [...], "hard": [...], "junk": [...]}


def generate(out_root: str, seed: int) -> Plan:
    ds_dir = os.path.join(out_root, DATASET)
    jpg_dir = os.path.join(ds_dir, "jpg")
    gt_dir = os.path.join(ds_dir, "gt")
    rv_dir = os.path.join(out_root, "revisitop")
    pkl_path = os.path.join(rv_dir, f"gnd_r{DATASET}.pkl")

    # Delete and recreate only this generator's own outputs.
    if os.path.isdir(ds_dir):
        shutil.rmtree(ds_dir)
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    os.makedirs(jpg_dir); os.makedirs(gt_dir); os.makedirs(rv_dir, exist_ok=True)

    plan = Plan()
    counter = 0

    def new_id(tag):
        nonlocal counter
        counter += 1
        return f"{DATASET}_{tag}_{counter:06d}"

    def save(img, iid):
        assert img.size == (WIDTH, HEIGHT) and img.mode == "RGB"
        img.save(os.path.join(jpg_dir, iid + ".jpg"), "JPEG", quality=JPEG_QUALITY)

    for li, name in enumerate(LANDMARKS):
        canon = render_landmark(li, name, seed)
        rng = np.random.default_rng([seed, li])
        grp = {"easy": [], "hard": [], "junk": []}
        for n in range(1, N_QUERY + 1):
            qid = new_id(name)
            img, bbox = make_view(canon, LEVELS["query"], rng)
            save(img, qid)
            plan.queries.append((qid, name, n, bbox))
        for kind, count in (("easy", N_EASY), ("hard", N_HARD), ("junk", N_JUNK)):
            for _ in range(count):
                iid = new_id(name)
                img, _ = make_view(canon, LEVELS[kind], rng)
                save(img, iid)
                grp[kind].append(iid)
                plan.db.append(iid)
        plan.groups[name] = grp

    rng = np.random.default_rng([seed, 999])
    for _ in range(N_NEGATIVE):
        iid = new_id(NEGATIVE_NAME)
        save(make_negative(rng), iid)
        plan.db.append(iid)

    # ---- Original (Oxford-style) text ground truth ----
    for qid, name, n, bbox in plan.queries:
        g = plan.groups[name]
        base = os.path.join(gt_dir, f"{name}_{n}")
        with open(base + "_query.txt", "w") as f:
            f.write(f"oxc1_{qid} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        for suffix, ids in (("good", g["easy"]), ("ok", g["hard"]), ("junk", g["junk"])):
            with open(f"{base}_{suffix}.txt", "w") as f:
                f.write("".join(i + "\n" for i in ids))

    # ---- Revisited (RevisitOP-style) pickle ground truth ----
    pos = {iid: k for k, iid in enumerate(plan.db)}
    gnd = []
    for qid, name, n, bbox in plan.queries:
        g = plan.groups[name]
        gnd.append({
            "bbx": [float(v) for v in bbox],
            "easy": [pos[i] for i in g["easy"]],
            "hard": [pos[i] for i in g["hard"]],
            "junk": [pos[i] for i in g["junk"]],
        })
    cfg = {"imlist": list(plan.db), "qimlist": [q[0] for q in plan.queries], "gnd": gnd}
    with open(pkl_path, "wb") as f:
        pickle.dump(cfg, f, protocol=4)
    return plan


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
class ValidationError(RuntimeError):
    pass


def validate(out_root: str) -> dict:
    errors: list[str] = []
    jpg_dir = os.path.join(out_root, DATASET, "jpg")
    gt_dir = os.path.join(out_root, DATASET, "gt")
    pkl_path = os.path.join(out_root, "revisitop", f"gnd_r{DATASET}.pkl")

    def err(msg):
        errors.append(msg)

    for p in (jpg_dir, gt_dir, pkl_path):
        if not os.path.exists(p):
            raise ValidationError(f"missing path: {p}")

    # 1. Every JPEG opens, is RGB and 512x384; no nested dirs.
    existing = set()
    for fn in sorted(os.listdir(jpg_dir)):
        path = os.path.join(jpg_dir, fn)
        if os.path.isdir(path):
            err(f"nested directory not allowed: {path}"); continue
        if not fn.endswith(".jpg"):
            err(f"unexpected file: {fn}"); continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                if im.format != "JPEG": err(f"{fn}: not JPEG ({im.format})")
                if im.mode != "RGB": err(f"{fn}: mode {im.mode} != RGB")
                if im.size != (WIDTH, HEIGHT): err(f"{fn}: size {im.size} != {(WIDTH, HEIGHT)}")
        except Exception as e:  # noqa: BLE001
            err(f"{fn}: cannot open ({e})"); continue
        existing.add(fn[:-4])
    if len(existing) < 150:
        err(f"only {len(existing)} images (< 150)")

    # 2. Pickle structure.
    with open(pkl_path, "rb") as f:
        cfg = pickle.load(f)
    for key in ("imlist", "qimlist", "gnd"):
        if key not in cfg:
            raise ValidationError(f"pickle missing key '{key}'")
    imlist, qimlist, gnd = cfg["imlist"], cfg["qimlist"], cfg["gnd"]
    if len(gnd) != len(qimlist):
        err(f"len(gnd)={len(gnd)} != len(qimlist)={len(qimlist)}")
    if len(set(imlist)) != len(imlist): err("duplicate ids in imlist")
    if len(set(qimlist)) != len(qimlist): err("duplicate ids in qimlist")
    if set(imlist) & set(qimlist): err("query ids appear in imlist")
    for iid in list(imlist) + list(qimlist):
        if iid not in existing: err(f"referenced image missing: {iid}")
    negatives = {i for i in imlist if f"_{NEGATIVE_NAME}_" in i}

    def check_bbox(b, where):
        if len(b) != 4: err(f"{where}: bbox must have 4 values"); return
        x1, y1, x2, y2 = b
        if not (0 <= x1 < x2 <= WIDTH - 1 and 0 <= y1 < y2 <= HEIGHT - 1):
            err(f"{where}: invalid bbox {b}")

    for qi, g in enumerate(gnd):
        for key in ("bbx", "easy", "hard", "junk"):
            if key not in g: err(f"gnd[{qi}] missing '{key}'")
        if any(k not in g for k in ("bbx", "easy", "hard", "junk")): continue
        check_bbox(g["bbx"], f"gnd[{qi}]")
        for key in ("easy", "hard", "junk"):
            for ix in g[key]:
                if not isinstance(ix, int) or not (0 <= ix < len(imlist)):
                    err(f"gnd[{qi}].{key}: invalid index {ix}")
        e, h, j = set(g["easy"]), set(g["hard"]), set(g["junk"])
        if e & h or e & j or h & j: err(f"gnd[{qi}]: easy/hard/junk not disjoint")
        if len(e) == 0: err(f"gnd[{qi}]: no easy positives")
        for ix in e | h:
            if 0 <= ix < len(imlist) and imlist[ix] in negatives:
                err(f"gnd[{qi}]: negative image used as positive")

    # 3. Text ground truth consistent with pickle.
    text_queries = []
    for fn in sorted(os.listdir(gt_dir)):
        if not fn.endswith("_query.txt"): continue
        base = fn[: -len("_query.txt")]
        with open(os.path.join(gt_dir, fn)) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        if len(lines) != 1: err(f"{fn}: expected exactly one line"); continue
        parts = lines[0].split()
        if len(parts) != 5 or not parts[0].startswith("oxc1_"):
            err(f"{fn}: malformed line '{lines[0]}'"); continue
        qid = parts[0][len("oxc1_"):]
        try:
            bb = [int(float(v)) for v in parts[1:]]
        except ValueError:
            err(f"{fn}: non-numeric bbox"); continue
        check_bbox(bb, fn)
        if qid not in existing: err(f"{fn}: query image missing {qid}")
        text_queries.append(qid)
        lists = {}
        for suffix in ("good", "ok", "junk"):
            p = os.path.join(gt_dir, f"{base}_{suffix}.txt")
            if not os.path.exists(p): err(f"missing {base}_{suffix}.txt"); lists[suffix] = set(); continue
            with open(p) as f:
                ids = [l.strip() for l in f if l.strip()]
            for iid in ids:
                if iid not in existing: err(f"{base}_{suffix}.txt: missing image {iid}")
                if iid == qid: err(f"{base}_{suffix}.txt contains its own query id")
                if suffix in ("good", "ok") and iid in negatives:
                    err(f"{base}_{suffix}.txt contains negative image {iid}")
            lists[suffix] = set(ids)
        if qid in qimlist:
            g = gnd[qimlist.index(qid)]
            if {imlist[i] for i in g["easy"]} != lists["good"]: err(f"{base}: good != pickle easy")
            if {imlist[i] for i in g["hard"]} != lists["ok"]: err(f"{base}: ok != pickle hard")
            if {imlist[i] for i in g["junk"]} != lists["junk"]: err(f"{base}: junk != pickle junk")
            if [int(v) for v in g["bbx"]] != bb: err(f"{base}: bbox differs from pickle")
    if sorted(text_queries) != sorted(qimlist):
        err("qimlist does not match text ground-truth query ids")

    if errors:
        shown = "\n  - ".join(errors[:50])
        more = f"\n  ... and {len(errors) - 50} more" if len(errors) > 50 else ""
        raise ValidationError(f"{len(errors)} validation error(s):\n  - {shown}{more}")

    return {"images": len(existing), "queries": len(qimlist), "database": len(imlist),
            "negatives": len(negatives)}


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic toy_a retrieval dataset.")
    ap.add_argument("--output", default="data", help="output root directory (default: data)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    ap.add_argument("--validate-only", action="store_true", help="only validate an existing dataset")
    args = ap.parse_args(argv)

    if not args.validate_only:
        print(f"Generating {DATASET} into '{args.output}' with seed {args.seed} ...")
        generate(args.output, args.seed)
    try:
        stats = validate(args.output)
    except ValidationError as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        return 1
    print("Validation passed.")
    print(f"  total images   : {stats['images']}")
    print(f"  query images   : {stats['queries']}")
    print(f"  database images: {stats['database']} (incl. {stats['negatives']} negatives)")
    print(f"  per landmark   : {N_QUERY} query / {N_EASY} easy / {N_HARD} hard / {N_JUNK} junk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
