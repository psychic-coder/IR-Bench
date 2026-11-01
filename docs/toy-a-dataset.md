# toy_a — Synthetic Image-Retrieval Dataset

`toy_a` is a small, **fully synthetic** landmark-retrieval dataset. It uses the same layout as the
Oxford/Paris benchmarks (text ground truth) and the RevisitOP benchmarks (pickle ground truth).
Use it to smoke-test retrieval pipelines, data loaders and evaluation code without downloading
real data.

> **Important:** every image is procedurally drawn with Pillow and NumPy. The "landmarks" are
> geometric drawings, not photographs. Scores such as mAP on `toy_a` show only whether a pipeline runs
> end-to-end and is roughly working. **They do not mean anything for real-world benchmarks** and
> must not be compared with or reported as results on Oxford5k, Paris6k, ROxford or RParis.

## Requirements

- Python 3.10+
- Pillow
- NumPy

The generator needs no internet, external datasets or pretrained models.

## Generation

```bash
python generate_toy_a.py --output data --seed 42
```

| Option            | Default | Description                                  |
|-------------------|---------|----------------------------------------------|
| `--output`        | `data`  | Output root directory                        |
| `--seed`          | `42`    | Random seed; the same seed gives identical files |
| `--validate-only` | off     | Validate an existing dataset without regenerating it |

Each run deletes and recreates only the generator's own outputs: `<output>/toy_a/` and
`<output>/revisitop/gnd_rtoy_a.pkl`. Other files under `<output>/revisitop/` are left alone.
The run then validates everything. If validation fails, it exits with code `1` and lists the
errors. If it succeeds, it prints the image and query counts.

## Output paths

```
data/
├── toy_a/
│   ├── jpg/                 # 222 images: <image_id>.jpg (flat, no subfolders)
│   └── gt/                  # 64 queries × 4 files = 256 text files
│       ├── tower_1_query.txt
│       ├── tower_1_good.txt
│       ├── tower_1_ok.txt
│       ├── tower_1_junk.txt
│       └── ...              # tower..hall, numbers 1..8
└── revisitop/
    └── gnd_rtoy_a.pkl
```

Image IDs look like `toy_a_<landmark>_<NNNNNN>`, for example `toy_a_tower_000001`. Negative images
use `toy_a_negative_<NNNNNN>`. The six-digit counter is global and unique.

## Image counts

| Split                    | Per landmark | Total |
|--------------------------|-------------:|------:|
| Query                    | 8            | 64    |
| Easy (database)          | 8            | 64    |
| Hard (database)          | 5            | 40    |
| Junk (database)          | 3            | 24    |
| Unrelated negatives (db) | —            | 30    |
| **Database total**       |              | **158** |
| **All images**           |              | **222** |

The 8 landmarks are `tower, bridge, church, museum, palace, gate, arch, hall`, labelled
`LANDMARK_01` … `LANDMARK_08`. All images are 512×384 RGB JPEGs saved at quality 92.

### How images are made

Each landmark is drawn once as a 768×576 canonical scene, and every image is a warped view of that
scene. Each scene has its own color palette, background texture (grid, checker or diagonal
stripes), building shape, signature symbols (circles, polygons, crosses, bars) and a text label.

| Tier  | Rotation | Scale      | Translation | Bright/contrast | Blur ≤ | Noise σ ≤ | Occluders | Notes                      |
|-------|---------:|------------|------------:|----------------:|-------:|----------:|-----------|----------------------------|
| query | ±4°      | 0.95–1.05  | ±12 px      | ±8%             | 0.6    | 3         | 0         | near-canonical view        |
| easy  | ±8°      | 0.90–1.12  | ±25 px      | ±15%            | 1.0    | 5         | 0–1 small | close match                |
| hard  | ±30°     | 0.60–1.45  | ±70 px      | ±40%            | 2.5    | 14        | 1–2 large | 50% chance of a zoomed-in crop |
| junk  | ±45°     | 2.4–3.4 zoom | ±40 px    | ±45%            | 3.0    | 18        | 2–3 large | small fragment of the landmark only |

Negatives combine random gradients, blobs, polygons and strokes. They contain no landmark
structure and no labels.

## Original ground-truth format (`data/toy_a/gt/`)

Each query has four files named `<landmark>_<number>_<type>.txt`, where `number` runs from 1 to 8.

- `*_query.txt` contains one line: `oxc1_<query_id> x1 y1 x2 y2`

  ```
  oxc1_toy_a_tower_000001 68 38 462 355
  ```

  The box is the landmark region of the canonical scene after the query's warp, clipped to
  the image: `0 ≤ x1 < x2 ≤ 511` and `0 ≤ y1 < y2 ≤ 383`.
- `*_good.txt` lists the strongest matches, one image ID per line (the easy tier).
- `*_ok.txt` lists additional matches (the hard tier).
- `*_junk.txt` lists weak or partial matches (the junk tier).

These lists never contain the query itself. Negative images never appear in `good` or `ok`.
All queries of the same landmark share the same good, ok and junk lists.

## Revisited ground-truth format (`data/revisitop/gnd_rtoy_a.pkl`)

```python
import pickle
cfg = pickle.load(open("data/revisitop/gnd_rtoy_a.pkl", "rb"))
cfg["imlist"]   # list[str] — 158 database image IDs (no queries)
cfg["qimlist"]  # list[str] — 64 query image IDs, same order as the text GT
cfg["gnd"]      # list[dict], one per query:
                #   {"bbx": [x1, y1, x2, y2],     # floats
                #    "easy": [int, ...],          # indices into imlist (= good)
                #    "hard": [int, ...],          # indices into imlist (= ok)
                #    "junk": [int, ...]}          # indices into imlist (= junk)
```

Images are loaded as `data/toy_a/jpg/<id>.jpg`. The `easy`, `hard` and `junk` lists are disjoint.
Negatives are in `imlist` but in none of the lists, so they count as non-matches. Following the
RevisitOP convention:

- **Easy protocol:** positives are `easy`; `hard` and `junk` are ignored.
- **Medium protocol:** positives are `easy` and `hard`; `junk` is ignored.
- **Hard protocol:** positives are `hard`; `easy` and `junk` are ignored.

## Validation

```bash
python generate_toy_a.py --output data --validate-only
```

Validation checks that:

- every JPEG opens and is RGB 512×384, with no nested folders
- every query and database ID in either ground-truth format exists on disk
- the pickle has the keys `imlist`, `qimlist` and `gnd`
- all indexes are valid integers
- all bounding boxes are valid
- `easy`, `hard` and `junk` are disjoint
- no query ID appears in `imlist`, and no negative is listed as a positive
- the text and pickle ground truths agree on query IDs, boxes and lists

## Archive (optional)

```bash
tar -czf toy_a_dataset.tar.gz generate_toy_a.py README_TOY_A.md data/
# or
zip -qr toy_a_dataset.zip generate_toy_a.py README_TOY_A.md data/
```
