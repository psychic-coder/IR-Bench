# Contributing

## Ground rules

1. **No hand-written metrics.** Every number in documentation, tests' expected values excepted, must come from
   `results/` produced by the harness. The README results block is generated; never edit it by hand.
2. **Everything runs in Docker.** A change is done when `docker compose up --build`, `make test`, `make lint` and
   `make bench-ci` still work.
3. **Seed everything, pin everything.** New randomness goes through the global seed; new dependencies are pinned in
   `backend/requirements*.txt` and the lock files regenerated (command in `backend/requirements.lock`), or pinned
   exactly in `frontend/package.json` with `package-lock.json` updated.
4. **Keep the UI thin.** The frontend renders API data; benchmark logic belongs in the backend.

## Workflow

```bash
make env                 # .env with your uid/gid
make test                # backend pytest in the image + frontend test stage
make lint                # ruff check + ruff format --check + eslint/prettier/tsc
make bench-ci            # offline end-to-end benchmark on toy data
```

Without Docker (backend): install CPU torch from the PyTorch index, `pip install -r backend/requirements-dev.lock`,
`pip install -e backend`, then `ruff check . && ruff format --check . && (cd backend && pytest ../tests)`.
Frontend: `cd frontend && npm ci && npm run lint && npm run format:check && npm run typecheck && npm test`.

## Adding components

See README §12 for code: backbones register with `@register_extractor`, poolings extend `features/pooling.py`,
indexes implement `VectorIndex` and register with `register_index`, re-rankers implement `Reranker` and register
with `register_reranker`. Each new component needs:

- a unit test (shapes and a numerical property with a known answer),
- an entry in at least one benchmark YAML experiment,
- documentation in README §7 (theory) and §9 (configuration).

## Code style

Python: type hints, docstrings on public functions, Ruff (config in `ruff.toml`, line length 110).
TypeScript: strict mode, ESLint (`next/core-web-vitals`, `next/typescript`), Prettier.
Commit messages: imperative mood, one logical change per commit.
