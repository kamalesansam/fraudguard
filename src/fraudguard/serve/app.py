"""FastAPI inference service.

Startup fails closed: if the model's SHA-256 does not match the manifest, or
the artifact needs a type outside the allowlist, the process exits and the
Kubernetes readiness probe never goes green.
"""

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from fraudguard.evaluate import predict_scores
from fraudguard.integrity import load_verified
from fraudguard.serve.schemas import (
    BatchRequest,
    BatchResponse,
    ModelInfo,
    Prediction,
    PredictionResponse,
    Transaction,
)
from fraudguard.serve.security import KeyStore, RateLimiter

MAX_BODY_BYTES = 64 * 1024
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
}

log = logging.getLogger("fraudguard.api")


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    api_key_hashes: str
    rate_limit_per_minute: int
    enable_docs: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_dir=Path(os.environ.get("FRAUDGUARD_MODEL_DIR", "artifacts/model")),
            api_key_hashes=os.environ.get("FRAUDGUARD_API_KEY_HASHES", ""),
            rate_limit_per_minute=int(os.environ.get("FRAUDGUARD_RATE_LIMIT_PER_MINUTE", "300")),
            enable_docs=os.environ.get("FRAUDGUARD_ENABLE_DOCS", "false").lower() == "true",
        )


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "fraudguard_requests_total", "HTTP requests", ["path", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "fraudguard_request_latency_seconds",
            "Request latency",
            ["path"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry,
        )
        self.predictions = Counter(
            "fraudguard_predictions_total",
            "Predictions by decision",
            ["decision"],
            registry=self.registry,
        )
        self.scores = Histogram(
            "fraudguard_fraud_score",
            "Distribution of fraud scores (prediction drift signal)",
            buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
            registry=self.registry,
        )
        self.auth_failures = Counter(
            "fraudguard_auth_failures_total", "Rejected API keys", registry=self.registry
        )
        self.rate_limited = Counter(
            "fraudguard_rate_limited_total",
            "Requests rejected by the rate limiter",
            registry=self.registry,
        )
        self.model_info = Gauge(
            "fraudguard_model_info", "Loaded model", ["version", "sha256"], registry=self.registry
        )


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


def _configure_logging() -> None:
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False


def create_app(load_model: bool = True, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    keys = KeyStore.from_env_value(settings.api_key_hashes)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    metrics = Metrics()
    state: dict[str, Any] = {}
    _configure_logging()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if load_model:
            manifest = json.loads((settings.model_dir / "manifest.json").read_text())
            state["model"] = load_verified(
                settings.model_dir / manifest["model"]["file"], manifest["model"]["sha256"]
            )
            state["manifest"] = manifest
            metrics.model_info.labels(manifest["version"], manifest["model"]["sha256"][:12]).set(1)
            log.info("model loaded", extra={"fields": {"sha256": manifest["model"]["sha256"]}})
        if not keys.digests:
            log.warning("no API keys configured: every request will be rejected")
        yield

    docs = settings.enable_docs
    app = FastAPI(
        title="FraudGuard",
        version="1.0.0",
        description="Fraud scoring API. All /v1 routes require an X-API-Key header.",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            response: Response = JSONResponse({"detail": "request body too large"}, status_code=413)
        else:
            start = time.perf_counter()
            response = await call_next(request)
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            metrics.latency.labels(path).observe(time.perf_counter() - start)
            metrics.requests.labels(path, str(response.status_code)).inc()
            log.info(
                "request",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": path,
                        "status": response.status_code,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                },
            )
        response.headers.update(SECURITY_HEADERS)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo the submitted value back: it may be NaN (not valid JSON) or
        # attacker-controlled content that ends up in someone's log viewer.
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return JSONResponse({"detail": errors}, status_code=422)

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authenticate(api_key: Annotated[str | None, Depends(api_key_header)]) -> str:
        client = keys.identify(api_key)
        if client is None:
            metrics.auth_failures.inc()
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
        allowed, retry_after = limiter.allow(client)
        if not allowed:
            metrics.rate_limited.inc()
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "rate limit exceeded",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        return client

    def require_model() -> tuple[Any, dict[str, Any]]:
        if "model" not in state:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return state["model"], state["manifest"]

    Client = Annotated[str, Depends(authenticate)]
    Loaded = Annotated[tuple[Any, dict[str, Any]], Depends(require_model)]

    def score(model: Any, manifest: dict[str, Any], txs: list[Transaction]) -> list[Prediction]:
        frame = pd.DataFrame([t.model_dump() for t in txs])
        threshold = manifest["model"]["threshold"]
        out = []
        for s in predict_scores(model, frame):
            flagged = bool(s >= threshold)
            metrics.scores.observe(float(s))
            metrics.predictions.labels("fraud" if flagged else "legit").inc()
            out.append(Prediction(fraud_score=round(float(s), 4), is_fraud=flagged))
        return out

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", include_in_schema=False)
    def ready() -> JSONResponse:
        ok = "model" in state
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(tx: Transaction, request: Request, _: Client, loaded: Loaded) -> PredictionResponse:
        model, manifest = loaded
        pred = score(model, manifest, [tx])[0]
        return PredictionResponse(
            **pred.model_dump(),
            request_id=request.state.request_id,
            threshold=manifest["model"]["threshold"],
            model_version=manifest["version"],
        )

    @app.post("/v1/predict/batch", response_model=BatchResponse)
    def predict_batch(
        body: BatchRequest, request: Request, _: Client, loaded: Loaded
    ) -> BatchResponse:
        model, manifest = loaded
        return BatchResponse(
            request_id=request.state.request_id,
            threshold=manifest["model"]["threshold"],
            model_version=manifest["version"],
            predictions=score(model, manifest, body.transactions),
        )

    @app.get("/v1/model", response_model=ModelInfo)
    def model_info(_: Client, loaded: Loaded) -> ModelInfo:
        _, manifest = loaded
        m = manifest["metrics"]
        return ModelInfo(
            version=manifest["version"],
            sha256=manifest["model"]["sha256"],
            threshold=manifest["model"]["threshold"],
            created_at=manifest["created_at"],
            git_commit=manifest["provenance"]["git_commit"],
            metrics={k: m[k] for k in ("roc_auc", "pr_auc", "precision", "recall")},
        )

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("FRAUDGUARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("FRAUDGUARD_PORT", "8000")),
        proxy_headers=False,
        server_header=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
