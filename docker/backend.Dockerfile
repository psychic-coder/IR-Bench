# syntax=docker/dockerfile:1.7
# Backend image used by the `api` and `benchmark` services.
#   TORCH_VARIANT=cpu   (default) CPU wheels from the PyTorch index
#   TORCH_VARIANT=cu121 CUDA 12.1 wheels (see docker-compose.gpu.yml)
ARG PYTHON_IMAGE=python:3.11.11-slim-bookworm

# ---------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
ARG TORCH_VARIANT=cpu
ARG INSTALL_DEV=true
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY backend/requirements.lock backend/requirements-dev.lock /tmp/req/
# torch/torchvision come from the PyTorch index for the chosen variant. The CPU wheels need
# no extra packages (--no-deps; their pure-Python deps are pinned in the lock files); CUDA
# wheels pull their nvidia-* runtime libraries. Everything else is installed from the fully
# pinned lock file, then `pip check` verifies the environment is consistent.
RUN if [ "$TORCH_VARIANT" = "cpu" ]; then NODEPS="--no-deps"; else NODEPS=""; fi \
    && pip install $NODEPS --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
        torch==2.5.1 torchvision==0.20.1 \
    && if [ "$INSTALL_DEV" = "true" ]; then LOCK=/tmp/req/requirements-dev.lock; else LOCK=/tmp/req/requirements.lock; fi \
    && pip install -r "$LOCK" \
    && pip check
COPY backend/pyproject.toml /src/backend/pyproject.toml
COPY backend/src /src/backend/src
RUN pip install --no-deps /src/backend

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --home-dir /tmp/home app \
    && mkdir -p /data /results /workspace /tmp/home \
    && chown app:app /data /results /workspace /tmp/home \
    && chmod 1777 /tmp/home
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY ruff.toml /app/ruff.toml
COPY backend/src /app/backend/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY tests /app/tests
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp/home \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    CONFIG_DIR=/app/configs \
    README_PATH=/workspace/README.md \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache/xdg \
    MPLCONFIGDIR=/tmp/matplotlib
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["irbench", "serve", "--host", "0.0.0.0", "--port", "8000"]
