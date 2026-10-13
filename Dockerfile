# Inference image. Contains the API, its runtime dependencies and the exact
# model artifact that passed the quality gate. No compilers, no dev tools,
# no training stack, runs as an unprivileged user.
# Base images are pinned by digest; Dependabot proposes updates.

FROM ghcr.io/astral-sh/uv:0.11.25@sha256:1e3808aa9023d0980e7c15b1fa7c1ac16ff35925780cf5c459858b2d693f01a9 AS uv

FROM python:3.14-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first so this layer is cached across code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serve --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra serve --no-editable

FROM python:3.14-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83 AS runtime
LABEL org.opencontainers.image.title="fraudguard" \
      org.opencontainers.image.description="Fraud scoring API with a verified model artifact" \
      org.opencontainers.image.source="https://github.com/kamalesansam/fraudguard" \
      org.opencontainers.image.licenses="MIT"
# No package managers at runtime: drop the base image's pip/setuptools/wheel
# (nothing to install malware with, and fewer CVEs to carry).
RUN python -m pip uninstall --yes --no-cache-dir pip setuptools wheel \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /opt/venv /opt/venv
# Model files stay root-owned and read-only for the app user: a compromised
# process cannot swap the model it serves.
COPY artifacts/model/model.skops artifacts/model/manifest.json artifacts/model/MODEL_CARD.md /app/model/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUDGUARD_MODEL_DIR=/app/model \
    FRAUDGUARD_HOST=0.0.0.0 \
    FRAUDGUARD_PORT=8000
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]
ENTRYPOINT ["fraudguard-serve"]
