# Convenience targets. Everything runs in Docker; only `docker compose` is required.
COMPOSE    ?= docker compose
GPU_FILES  := -f docker-compose.yml -f docker-compose.gpu.yml
SUBSET     ?= 300
API_PORT   ?= 8000
UI_PORT    ?= 3000
CONFIG     ?= full
ABL        ?= c
ABL_SUBSET ?=

.DEFAULT_GOAL := help
.PHONY: help up up-gpu down logs ps bench bench-fast bench-ci bench-ablation bench-gpu report loadtest test test-backend test-frontend lint clean env

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env: ## create .env from .env.example (keeps an existing .env) and set APP_UID/APP_GID
	@test -f .env || (cp .env.example .env && sed -i.bak "s/^APP_UID=.*/APP_UID=$$(id -u)/; s/^APP_GID=.*/APP_GID=$$(id -g)/" .env && rm -f .env.bak && echo "created .env")

up: env ## build and start api, ui, redis, qdrant (detached)
	$(COMPOSE) up --build -d
	@echo "UI:  http://localhost:$(UI_PORT)    API: http://localhost:$(API_PORT)/docs    readiness: http://localhost:$(API_PORT)/ready"

up-gpu: env ## same as up, with the NVIDIA override
	$(COMPOSE) $(GPU_FILES) up --build -d

down: ## stop all services (keeps volumes)
	$(COMPOSE) --profile benchmark --profile loadtest down --remove-orphans

logs: ## follow api + ui logs
	$(COMPOSE) logs -f api ui

ps: ## service status and health
	$(COMPOSE) ps

bench: env ## full benchmark (configs/full.yaml) -> ./results
	BENCH_CONFIG=full DATASET_SUBSET= $(COMPOSE) --profile benchmark run --rm --build benchmark

bench-fast: env ## smoke benchmark (configs/fast.yaml, DATASET_SUBSET=$(SUBSET)) -> ./results
	BENCH_CONFIG=fast DATASET_SUBSET=$(SUBSET) $(COMPOSE) --profile benchmark run --rm --build benchmark

bench-ci: env ## offline toy-data benchmark (configs/ci.yaml), no downloads
	BENCH_CONFIG=ci DATASET_SUBSET= $(COMPOSE) --profile benchmark run --rm --build benchmark

bench-ablation: env ## one ablation only: make bench-ablation ABL=c [CONFIG=fast] [ABL_SUBSET=300]
	DATASET_SUBSET=$(ABL_SUBSET) $(COMPOSE) --profile benchmark run --rm --build benchmark \
		irbench bench --config $(CONFIG).yaml --ablation $(ABL)

bench-gpu: env ## full benchmark on an NVIDIA GPU
	BENCH_CONFIG=full DATASET_SUBSET= $(COMPOSE) $(GPU_FILES) --profile benchmark run --rm --build benchmark

report: ## rebuild tables/plots/summary/README from results/runs.jsonl (BENCH_CONFIG=fast|full)
	$(COMPOSE) --profile benchmark run --rm --no-deps benchmark irbench report --config $${BENCH_CONFIG:-full}.yaml

loadtest: env ## k6 load test against the running api -> ./results/loadtest
	$(COMPOSE) --profile loadtest run --rm loadtest

test: test-backend test-frontend ## all tests

test-backend: env ## pytest inside the backend image
	$(COMPOSE) build api
	$(COMPOSE) run --rm --no-deps -e QDRANT_URL=:memory: -e DATA_DIR=/tmp/irbench-test-data api \
		python -m pytest -p no:cacheprovider -c /app/backend/pyproject.toml --rootdir /app /app/tests -q

test-frontend: ## lint + typecheck + vitest inside the frontend test stage
	docker build -f docker/frontend.Dockerfile --target test .

lint: env ## ruff (check + format check) and the frontend lint stage
	$(COMPOSE) build api
	$(COMPOSE) run --rm --no-deps api sh -c "cd /app && ruff check --no-cache . && ruff format --no-cache --check ."
	docker build -f docker/frontend.Dockerfile --target test .

clean: ## remove containers, volumes, results and derived caches (keeps downloaded datasets)
	$(COMPOSE) --profile benchmark --profile loadtest down -v --remove-orphans
	find results -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
	rm -rf data/cache data/toy_a data/toy_b data/toy_distractors
