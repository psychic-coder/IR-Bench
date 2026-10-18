"""Command-line entry point: ``python -m irbench.cli <command>``.

Commands
--------
bench          run a benchmark YAML, write results/ and refresh the README block
report         regenerate tables/plots/summary from results/runs.jsonl (no re-running)
download       fetch datasets (and optionally distractor archives) into DATA_DIR
toy            generate the synthetic toy datasets
update-readme  refresh the README results block from results/summary.md
serve          start the FastAPI app with uvicorn
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from irbench.logging_utils import configure_logging, get_logger
from irbench.settings import get_settings

log = get_logger("irbench.cli")


def _resolve_config(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    candidate = get_settings().config_dir / path
    if candidate.exists():
        return candidate
    candidate = get_settings().config_dir / f"{path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"benchmark config '{path}' not found")


def _filter_experiments(cfg, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """Keep only experiments selected with --ablation / --experiment (all when neither is given)."""
    ablations = set(args.ablation or [])
    names = set(args.experiment or [])
    if not ablations and not names:
        return cfg
    kept = [e for e in cfg.experiments if e.ablation in ablations or e.name in names]
    if not kept:
        raise SystemExit(
            f"no experiment matches --ablation {sorted(ablations)} / --experiment {sorted(names)}"
        )
    return cfg.model_copy(update={"experiments": kept})


def _load_config(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from irbench.bench import load_benchmark_config

    cfg = load_benchmark_config(_resolve_config(args.config), get_settings().dataset_subset)
    return _filter_experiments(cfg, args)


def cmd_bench(args: argparse.Namespace) -> int:
    from irbench.bench import BenchmarkRunner
    from irbench.bench.report import write_report
    from irbench.data import DatasetUnavailableError

    settings = get_settings()
    cfg = _load_config(args)
    results_dir = Path(args.results_dir) if args.results_dir else settings.results_dir
    runner = BenchmarkRunner(
        cfg, settings, results_dir, resume=not args.no_resume, show_progress=not args.quiet
    )
    try:
        records, meta = runner.run_all()
    except DatasetUnavailableError as exc:
        print("\n" + "=" * 78 + "\nDATASET NOT AVAILABLE\n" + "=" * 78 + f"\n{exc}\n", file=sys.stderr)
        return 2
    paths = write_report(
        records, meta, results_dir, runner.dataset, cfg.hard_query_ap_threshold, cfg.hard_query_top
    )
    log.info("results written", extra={k: str(v) for k, v in paths.items()})
    if not args.no_readme:
        _update_readme(settings.readme_path, results_dir)
    n_failed = sum(r.get("status") == "failed" for r in records)
    return 1 if (n_failed and args.strict) else 0


def cmd_report(args: argparse.Namespace) -> int:
    from irbench.bench import BenchmarkRunner
    from irbench.bench.report import write_report

    settings = get_settings()
    cfg = _load_config(args)
    results_dir = Path(args.results_dir) if args.results_dir else settings.results_dir
    runner = BenchmarkRunner(cfg, settings, results_dir, resume=True, show_progress=False)
    done = runner.store.load()
    records = []
    for p in runner.plan():
        if p.run_id in done:
            rec = dict(done[p.run_id])
            rec["experiments"] = [{"name": n, "ablation": a} for n, a in p.experiments]
            records.append(rec)
    for ds in sorted({r["dataset"] for r in records}):
        runner.dataset(ds)
    meta = runner.meta(0.0)
    previous = results_dir / "results.json"
    meta["wall_time_s"] = None  # unknown unless a previous full report recorded it
    if previous.exists():
        with contextlib.suppress(OSError, ValueError, KeyError):
            meta["wall_time_s"] = json.loads(previous.read_text(encoding="utf-8"))["meta"].get("wall_time_s")
    meta["report_regenerated"] = True
    write_report(records, meta, results_dir, runner.dataset, cfg.hard_query_ap_threshold, cfg.hard_query_top)
    if not args.no_readme:
        _update_readme(settings.readme_path, results_dir)
    return 0


def _update_readme(readme: Path, results_dir: Path) -> None:
    from irbench.readme import update_readme

    if not readme.exists():
        log.info("README not mounted; skipping README update", extra={"readme": str(readme)})
        return
    try:
        changed = update_readme(readme, results_dir / "summary.md", results_dir / "results.json")
        log.info("README results block refreshed", extra={"changed": changed})
    except (OSError, ValueError) as exc:
        log.warning("could not update README", extra={"error": str(exc)})


def cmd_download(args: argparse.Namespace) -> int:
    from irbench.data import DatasetUnavailableError, load_dataset
    from irbench.data.download import ensure_distractors

    settings = get_settings()
    rc = 0
    for name in args.datasets:
        try:
            ds = load_dataset(name, settings.data_dir, auto_download=settings.auto_download)
            print(f"{name}: OK {ds.describe()['protocols']} (missing/corrupt: {len(ds.missing)})")
        except DatasetUnavailableError as exc:
            print(f"\n{name}: NOT AVAILABLE\n{exc}\n", file=sys.stderr)
            rc = 2
    if args.distractor_archives:
        ensure_distractors(
            settings.data_dir,
            settings.resolved_distractor_dir,
            args.distractor_archives,
            settings.auto_download,
        )
    return rc


def cmd_toy(args: argparse.Namespace) -> int:
    from irbench.data.toy import TOY_DATASETS, generate_toy_dataset

    settings = get_settings()
    for name in TOY_DATASETS:
        print(generate_toy_dataset(name, settings.data_dir, seed=args.seed))
    return 0


def cmd_update_readme(args: argparse.Namespace) -> int:
    from irbench.readme import update_readme

    results = Path(args.results_dir)
    changed = update_readme(Path(args.readme), results / "summary.md", results / "results.json")
    print("README updated" if changed else "README already up to date")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "irbench.api.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_config=None,
        access_log=False,
        workers=1,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="irbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run a benchmark config")
    b.add_argument("--config", default="fast.yaml")
    b.add_argument("--results-dir", default=None)
    b.add_argument("--no-resume", action="store_true", help="discard previous runs.jsonl")
    b.add_argument("--no-readme", action="store_true")
    b.add_argument("--strict", action="store_true", help="exit 1 if any run failed")
    b.add_argument("--quiet", action="store_true", help="no progress bars")
    b.add_argument(
        "--ablation", action="append", help="only experiments of this ablation letter (repeatable)"
    )
    b.add_argument("--experiment", action="append", help="only the named experiment (repeatable)")
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("report", help="regenerate reports from stored runs")
    r.add_argument("--config", default="fast.yaml")
    r.add_argument("--results-dir", default=None)
    r.add_argument("--no-readme", action="store_true")
    r.add_argument("--ablation", action="append", help="report only experiments of this ablation letter")
    r.add_argument("--experiment", action="append", help="report only the named experiment")
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("download", help="download datasets")
    d.add_argument("--datasets", nargs="+", default=["oxford5k", "paris6k"])
    d.add_argument("--distractor-archives", type=int, default=0)
    d.set_defaults(func=cmd_download)

    t = sub.add_parser("toy", help="generate toy datasets")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_toy)

    u = sub.add_parser("update-readme", help="refresh README results block")
    u.add_argument("--readme", default="README.md")
    u.add_argument("--results-dir", default="results")
    u.set_defaults(func=cmd_update_readme)

    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging(get_settings().log_level)
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
