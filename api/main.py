"""Single-process FastAPI adapter for the immutable frozen inference bundle."""

import asyncio
import json
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import ApiSettings, runtime_from_request
from api.errors import classify_input_error, http_error_handler, validation_error_handler
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResult,
    PredictionRequest,
    PredictionResult,
    ReadinessResult,
    VersionResult,
)
from fraud_detection.inference import FrozenFraudInferenceRuntime

LOGGER = logging.getLogger("fraud_detection.api")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
LOGGER.propagate = False
SERVICE_VERSION = "0.1.0"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _request_id(request: Request) -> str:
    incoming = request.headers.get("X-Request-ID", "")
    return incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex


def _response_row(row: pd.Series) -> PredictionResult:
    payload = row.to_dict()
    if "triggered_rule_ids" in payload:
        payload["triggered_rule_ids"] = (
            payload["triggered_rule_ids"].split("|") if payload["triggered_rule_ids"] else []
        )
    payload["TransactionID"] = int(payload["TransactionID"])
    return PredictionResult.model_validate(payload)


def create_app(
    settings: ApiSettings | None = None,
    *,
    loader: Callable[[Path], FrozenFraudInferenceRuntime] = FrozenFraudInferenceRuntime.load,
) -> FastAPI:
    """Create one application whose lifespan loads one frozen runtime instance."""
    configured = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = None
        application.state.ready = False
        application.state.inference_lock = asyncio.Lock()
        try:
            application.state.runtime = await asyncio.to_thread(loader, configured.bundle_path)
            application.state.ready = True
            history = application.state.runtime.feature_state.amount_engineer.history_
            application.state.last_key = (
                (float(history.TransactionDT.iloc[-1]), int(history.TransactionID.iloc[-1]))
                if history is not None and not history.empty
                else None
            )
            LOGGER.info(
                json.dumps(
                    {
                        "event": "bundle_loaded",
                        "bundle_version": application.state.runtime.manifest["bundle_version"],
                        "feature_count": application.state.runtime.manifest["feature_count"],
                    }
                )
            )
        except Exception:
            LOGGER.exception("Frozen bundle startup failed; service is not ready")
        yield
        application.state.ready = False
        application.state.runtime = None

    application = FastAPI(
        title="Frozen Fraud Inference",
        description=(
            "Local serving of the locked TRAIN-only fraud system. Calibrated probability is "
            "followed by Balanced decisioning; the frozen rules overlay is optional. "
            "Stateful requests must arrive in chronological order."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    application.state.settings = configured
    application.state.runtime = None
    application.state.ready = False
    application.add_exception_handler(HTTPException, http_error_handler)
    application.add_exception_handler(RequestValidationError, validation_error_handler)

    @application.middleware("http")
    async def correlation_id(request: Request, call_next):
        started = time.perf_counter()
        request.state.request_id = _request_id(request)
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception("Unhandled API request failure")
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {"code": "http_500", "message": "Internal server failure"},
                    "request_id": request.state.request_id,
                },
            )
        response.headers["X-Request-ID"] = request.state.request_id
        if request.url.path in {"/predict", "/predict/batch"}:
            runtime = runtime_from_request(request)
            LOGGER.info(
                json.dumps(
                    {
                        "request_id": request.state.request_id,
                        "endpoint": request.url.path,
                        "transaction_count": getattr(request.state, "transaction_count", None),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "bundle_version": (
                            runtime.manifest["bundle_version"] if runtime is not None else None
                        ),
                        "action_counts": getattr(request.state, "action_counts", {}),
                        "status_code": response.status_code,
                    }
                )
            )
        return response

    @application.get("/health", description="Lightweight process liveness; performs no prediction.")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get(
        "/ready",
        response_model=ReadinessResult,
        description="Ready only after frozen bundle loading and checksum verification succeed.",
    )
    async def ready(request: Request) -> ReadinessResult:
        runtime = runtime_from_request(request)
        if not request.app.state.ready or runtime is None:
            raise HTTPException(status_code=503, detail="Frozen inference runtime is not ready")
        return ReadinessResult(
            ready=True,
            bundle_version=runtime.manifest["bundle_version"],
            feature_count=runtime.manifest["feature_count"],
            rules_enabled=configured.enable_rules,
        )

    @application.get(
        "/version",
        response_model=VersionResult,
        description="Non-sensitive identifiers for the loaded frozen system.",
    )
    async def version(request: Request) -> VersionResult:
        runtime = runtime_from_request(request)
        if not request.app.state.ready or runtime is None:
            raise HTTPException(status_code=503, detail="Frozen inference runtime is not ready")
        return VersionResult(
            service_version=SERVICE_VERSION,
            bundle_version=runtime.manifest["bundle_version"],
            feature_count=runtime.manifest["feature_count"],
            ensemble="0.8 LightGBM + 0.2 XGBoost",
            calibration_method="Platt",
            primary_policy="Balanced expected-cost",
            rules_enabled=configured.enable_rules,
        )

    async def score(
        request: Request, transactions: list[dict], include_rules: bool
    ) -> pd.DataFrame:
        runtime = runtime_from_request(request)
        if not request.app.state.ready or runtime is None:
            raise HTTPException(status_code=503, detail="Frozen inference runtime is not ready")
        request.state.transaction_count = len(transactions)
        try:
            async with request.app.state.inference_lock:
                first = transactions[0]
                try:
                    first_key = (float(first["TransactionDT"]), int(first["TransactionID"]))
                except (KeyError, TypeError, ValueError):
                    first_key = None  # The frozen input contract will report the schema error.
                last_key = request.app.state.last_key
                if first_key is not None and last_key is not None and first_key <= last_key:
                    raise ValueError("Stateful transactions must follow prior TransactionDT and ID")
                snapshot = await asyncio.to_thread(runtime.snapshot_state)
                try:
                    result = await asyncio.to_thread(
                        runtime.predict_batch, transactions, include_rules=include_rules
                    )
                except Exception:
                    await asyncio.to_thread(runtime.restore_state, snapshot)
                    raise
                last = transactions[-1]
                request.app.state.last_key = (
                    float(last["TransactionDT"]),
                    int(last["TransactionID"]),
                )
            column = "final_decision" if include_rules else "base_decision"
            request.state.action_counts = dict(Counter(result[column].tolist()))
            return result
        except ValueError as error:
            client_error = classify_input_error(error)
            raise client_error from error
        except Exception as error:
            LOGGER.exception("Frozen inference request failed")
            raise HTTPException(status_code=500, detail="Internal inference failure") from error

    @application.post(
        "/predict",
        response_model=PredictionResult,
        response_model_exclude_none=True,
        description=(
            "Score one label-free transaction. Returns calibrated fraud probability, the "
            "Balanced base decision, and optional frozen-rule action. Updates process-local "
            "non-label history; transactions must follow chronological order."
        ),
    )
    async def predict(request: Request, body: PredictionRequest) -> PredictionResult:
        include_rules = (
            configured.enable_rules if body.include_rules is None else body.include_rules
        )
        result = await score(request, [body.transaction], include_rules)
        return _response_row(result.iloc[0])

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResult,
        response_model_exclude_none=True,
        description=(
            "Score a non-empty chronological batch without reordering rows. Calibrated "
            "probability feeds Balanced decisioning; frozen rules are an optional overlay. "
            "The batch updates process-local non-label history atomically."
        ),
    )
    async def predict_batch(
        request: Request, body: BatchPredictionRequest
    ) -> BatchPredictionResult:
        request.state.transaction_count = len(body.transactions)
        if not body.transactions:
            raise HTTPException(
                status_code=422, detail="Batch must contain at least one transaction"
            )
        if len(body.transactions) > configured.max_batch_size:
            raise HTTPException(
                status_code=422,
                detail=f"Batch exceeds configured maximum of {configured.max_batch_size}",
            )
        include_rules = (
            configured.enable_rules if body.include_rules is None else body.include_rules
        )
        result = await score(request, body.transactions, include_rules)
        predictions = [_response_row(row) for _, row in result.iterrows()]
        return BatchPredictionResult(
            predictions=predictions,
            count=len(predictions),
            bundle_version=result.bundle_version.iloc[0],
        )

    return application


app = create_app()
