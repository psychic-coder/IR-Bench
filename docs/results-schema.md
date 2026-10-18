# Results file schemas

All files are written by the benchmark harness (`irbench bench`) or k6. Numbers are measurements; `null` means
"not applicable" (e.g. ANN recall for a flat index) or "undefined" (e.g. mAP when no query has a positive).

## `results/runs.jsonl`

One JSON object per completed run, appended as soon as the run finishes (crash-safe; later lines override earlier
lines with the same `run_id`). Used for resuming. Same record format as `results.json → runs[]` minus the
`experiments` field, plus `traceback` for failed runs.

## `results/results.json`

```text
{
  "meta": {
    "schema_version": int,                 # 3
    "benchmark": str,                      # config name (fast | full | ci | ...)
    "config_hash": str,                    # hash of the whole benchmark YAML after env overrides
    "config": {...},                       # the full benchmark configuration
    "seed": int,
    "subset": int | null,                  # DATASET_SUBSET
    "device": "cpu" | "cuda",
    "hardware": {platform, python, cpu_model, cpu_logical_cores, device, ram_gb, torch_threads, gpu_name?, cuda_version?},
    "libraries": {package: version},
    "datasets": {name: {n_images_pool, n_missing_or_corrupt, missing_examples, protocols: {p: {n_db, n_queries}},
                        n_corrupt, corrupt_ids, <protocol>_db_dropped, <protocol>_queries_dropped,
                        n_missing_referenced, subset, heldout}},
    "distractors": {"real" | "toy": {source, available, pool_size, skipped_corrupt, notes}},
    "headline": {protocol, setup},
    "ablations": {letter: description},
    "generated_at": ISO-8601 UTC,
    "wall_time_s": float,
    "timing_notes": str
  },
  "runs": [ run record ],
  "ablations": [ {key, title, columns, rows, notes, plot, extra} ]
}
```

### Run record

| field | type | meaning |
|---|---|---|
| `run_id` | str | hash of run config + dataset + protocol + subset + eval depth + seed + device + schema |
| `status` | `ok` / `skipped` / `failed` | skipped = invalid combination or missing resource; failed = exception |
| `reason` | str | why skipped/failed |
| `dataset`, `protocol` | str | e.g. `oxford5k`, `revisited` |
| `config` | object | the run configuration (extractor, pooling, postprocess, index, rerank, distractors) |
| `config_hash` | str | hash of `config` (same across datasets/protocols) |
| `label` | str | human-readable configuration |
| `family`, `extractor`, `pooling`, `index`, `rerank` | str | denormalised for filtering |
| `experiments` | list | `{name, ablation}` of every experiment that uses this run |
| `headline_map` | float | mAP (%) for the headline setup (`all` for the original protocol) |
| `metrics` | object | per setup (`easy`/`medium`/`hard` or `all`): `map`, `recall@1`, `recall@5`, `recall@10`, `n_valid_queries` (percent) |
| `per_query_ap` | object | per setup: list of AP in [0,1] (null = no positives), aligned with `query_ids` |
| `query_ids` | list | query image ids in evaluation order |
| `latency_ms` | object | `extract`, `search`, `rerank`, `total`: `{p50, p95, p99, mean, n}` |
| `throughput_qps` | object | `end_to_end_single_stream`, `search_batch` |
| `ann_recall@1`, `ann_recall@<k>`, `ann_recall_k` | float, int | neighbour recall vs exact search (approximate indexes only) |
| `ann_search_ms` | object | latency stats of the top-k searches used for neighbour recall |
| `memory` | object | `total_bytes`, `n_vectors`, `bytes_per_vector`, `encoding_bytes_per_vector`, `breakdown`, `method` (`exact`/`estimate`), `serialized_bytes`, `notes` |
| `vector_dim` | int | dimensionality after post-processing |
| `index_params_effective` | object | parameters actually used (after clamping) |
| `index_reused_from_previous_run` | bool | the build was shared with the previous run (search-parameter sweep) |
| `build_time_s` | float | index build time |
| `extract_ms_per_image_db` | float | batched database extraction time per image |
| `extract_db_from_cache` | bool | database descriptors came from the disk cache |
| `n_db`, `n_distractors`, `n_index`, `n_queries`, `eval_depth` | int | sizes |
| `pca` | object or null | `dim_requested`, `dim_effective`, `whiten`, `power`, `fit_on`, `n_fit`, `explained_variance_ratio` |
| `distractor_notes` | list | warnings about the distractor pool |
| `device` | str | device used |
| `extractor_config` | object | the descriptor configuration used for the cache key |
| `created_at` | str | ISO-8601 UTC |

## `results/results.csv`

One row per run with flattened columns: identifiers, `headline_map`, `map_easy`, `map_medium`, `map_hard`,
`map_original`, `recall@{1,5,10}` (Medium or original), `ann_recall@1`, `ann_recall@k`, `p{50,95,99}_{extract,search,rerank,total}_ms`,
`qps_single_stream`, `qps_search_batch`, `bytes_per_vector`, `encoding_bytes_per_vector`, `index_total_bytes`,
`memory_method`, `build_time_s`, `extract_ms_per_image_db`, `n_db`, `n_queries`, `config_hash`, `experiments`
(`letter:name;...`), `reason`.

## `results/rankings/<run_id>.json`

`{query_ids, top_ids: [[global ids]], top_scores: [[floats]]}` — the first 50 results of every query, used to build
`hard_queries.json`.

## `results/hard_queries.json`

```text
{
  "generated_at": str,
  "threshold_ap": float,                      # Medium AP threshold
  "datasets": {
    name: {
      "best_run_id": str, "best_config": str, "best_map_medium": float,
      "n_queries": int, "n_hard": int,
      "queries": [ {
        "query_id": "<dataset>/<image>",
        "bbox": [x1, y1, x2, y2] | null,
        "ap": {"easy": float|null, "medium": float|null, "hard": float|null},
        "n_positives": int, "n_junk": int,
        "expected_positives": [global ids, at most 60],
        "first_positive_rank_in_top50": int | null,
        "returned": [ {rank, id, score, label: positive|junk|negative|distractor} ]
      } ]
    }
  }
}
```

The best run per dataset is the successful revisited-protocol run without distractors with the highest headline mAP.

## `results/summary.md` and `results/plots/*.png`

Markdown rendering of the metadata, dataset statistics, headline table and every ablation table with its plot and
notes. It is the source of the README results block.

## `results/loadtest/summary-<timestamp>.json`, `summary-latest.json`

k6's standard end-of-test summary (`metrics`, `root_group`, `options`, `state`) plus a `meta` object with the load-test
configuration (`api`, `max_vus`, `duration_scale`, `cold_ratio`, `backbone`, `index`, `rerank`, `k`, `thresholds`).
Custom metrics: `server_total_ms`, `server_extract_ms`, `server_search_ms`, `server_rerank_ms` (trends, from the API
response), `embedding_cache_hits`, `embedding_cache_misses` (counters).
