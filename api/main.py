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
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import ROOT, ApiSettings, runtime_from_request
from api.errors import classify_input_error, http_error_handler, validation_error_handler
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResult,
    PredictionRequest,
    PredictionResult,
    ReadinessResult,
    StoredPredictionResult,
    VersionResult,
)
from fraud_detection.feedback import FeedbackService
from fraud_detection.feedback.schemas import (
    LabelBatchInput,
    LabelBatchResult,
    LabelInput,
    LabelResult,
    utc,
)
from fraud_detection.feedback.service import LabelConflict, UnknownPrediction
from fraud_detection.inference import FrozenFraudInferenceRuntime
from fraud_detection.monitoring.service import MonitoringService, choose_window
from fraud_detection.persistence import PersistenceStore
from fraud_detection.persistence.recovery import recover_runtime
from fraud_detection.persistence.service import (
    PersistenceService,
    PersistenceUnavailable,
    PredictionConflict,
)

LOGGER = logging.getLogger("fraud_detection.api")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
LOGGER.propagate = False
SERVICE_VERSION = "0.1.0"
FROM_QUERY = Query(default=None, alias="from")
TO_QUERY = Query(default=None, alias="to")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


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
    store: PersistenceStore | None = None,
) -> FastAPI:
    """Create one application whose lifespan loads one frozen runtime instance."""
    configured = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = None
        application.state.service = None
        application.state.ready = False
        application.state.database_ready = False
        application.state.state_recovered = False
        application.state.replayed_transaction_count = 0
        application.state.inference_lock = asyncio.Lock()
        try:
            application.state.runtime = await asyncio.to_thread(loader, configured.bundle_path)
            application.state.store = store or (
                PersistenceStore.from_url(configured.database_url)
                if configured.database_url
                else None
            )
            if application.state.store is None:
                raise RuntimeError("FRAUD_DATABASE_URL is required for durable inference")
            await asyncio.to_thread(application.state.store.check_ready)
            application.state.database_ready = True
            recovery = await asyncio.to_thread(
                recover_runtime, application.state.runtime, application.state.store
            )
            application.state.service = PersistenceService(
                application.state.runtime, application.state.store, recovery.last_key
            )
            application.state.feedback = FeedbackService(application.state.store)
            application.state.monitoring = MonitoringService(
                application.state.store, ROOT / "models/monitoring_v1"
            )
            application.state.replayed_transaction_count = recovery.transaction_count
            application.state.state_recovered = True
            application.state.ready = True
            LOGGER.info(
                json.dumps(
                    {
                        "event": "runtime_recovered",
                        "bundle_version": application.state.runtime.manifest["bundle_version"],
                        "feature_count": application.state.runtime.manifest["feature_count"],
                        "replayed_request_count": recovery.request_count,
                        "replayed_transaction_count": recovery.transaction_count,
                        "replay_duration_ms": round(recovery.duration_ms, 2),
                    }
                )
            )
        except Exception:
            LOGGER.exception("Frozen runtime or database recovery failed; service is not ready")
        yield
        application.state.ready = False
        application.state.runtime = None
        application.state.service = None

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
    application.state.service = None
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
                        "idempotent_replay": getattr(request.state, "idempotent_replay", False),
                        "validation_ms": getattr(request.state, "validation_ms", None),
                        "db_lookup_ms": getattr(request.state, "db_lookup_ms", None),
                        "inference_ms": getattr(request.state, "inference_ms", None),
                        "db_write_ms": getattr(request.state, "db_write_ms", None),
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
        description="Ready after bundle verification, database schema check, and runtime replay.",
    )
    async def ready(request: Request) -> ReadinessResult:
        runtime = runtime_from_request(request)
        if not request.app.state.ready or runtime is None:
            raise HTTPException(status_code=503, detail="Frozen inference runtime is not ready")
        try:
            await asyncio.to_thread(request.app.state.store.check_ready)
        except Exception as error:
            request.app.state.ready = False
            request.app.state.database_ready = False
            raise HTTPException(status_code=503, detail="Database is not ready") from error
        return ReadinessResult(
            ready=True,
            bundle_version=runtime.manifest["bundle_version"],
            feature_count=runtime.manifest["feature_count"],
            rules_enabled=configured.enable_rules,
            database_ready=True,
            state_recovered=request.app.state.state_recovered,
            replayed_transaction_count=request.app.state.replayed_transaction_count,
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
        request: Request, transactions: list[dict], include_rules: bool, endpoint_type: str
    ) -> pd.DataFrame:
        runtime = runtime_from_request(request)
        if not request.app.state.ready or runtime is None:
            raise HTTPException(status_code=503, detail="Frozen inference runtime is not ready")
        request.state.transaction_count = len(transactions)
        idempotency_key = request.headers.get("X-Idempotency-Key")
        if idempotency_key is not None and not IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key):
            raise HTTPException(status_code=422, detail="Invalid X-Idempotency-Key")
        try:
            async with request.app.state.inference_lock:
                result = await asyncio.to_thread(
                    request.app.state.service.score,
                    endpoint_type,
                    transactions,
                    rules_enabled=include_rules,
                    idempotency_key=idempotency_key,
                    correlation_id=request.state.request_id,
                )
            column = "final_decision" if include_rules else "base_decision"
            request.state.action_counts = dict(Counter(result.frame[column].tolist()))
            request.state.idempotent_replay = result.replayed
            request.state.validation_ms = round(result.validation_ms, 2)
            request.state.db_lookup_ms = round(result.db_lookup_ms, 2)
            request.state.inference_ms = round(result.inference_ms, 2)
            request.state.db_write_ms = round(result.db_write_ms, 2)
            return result.frame
        except PredictionConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            client_error = classify_input_error(error)
            raise client_error from error
        except PersistenceUnavailable as error:
            request.app.state.ready = False
            request.app.state.database_ready = False
            raise HTTPException(
                status_code=503, detail="Durable inference store is unavailable"
            ) from error
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
        result = await score(request, [body.transaction], include_rules, "predict")
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
        result = await score(request, body.transactions, include_rules, "predict_batch")
        predictions = [_response_row(row) for _, row in result.iterrows()]
        return BatchPredictionResult(
            predictions=predictions,
            count=len(predictions),
            bundle_version=result.bundle_version.iloc[0],
        )

    @application.get(
        "/predictions",
        description=(
            "List recent audit-safe prediction fields. Stored raw transaction payloads and "
            "feature vectors are never returned."
        ),
    )
    async def list_predictions(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0, le=100_000),
        from_time: datetime | None = FROM_QUERY,
        to_time: datetime | None = TO_QUERY,
        decision: str | None = None,
    ) -> dict:
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="Durable prediction store is not ready")
        if decision is not None and decision.upper() not in {"APPROVE", "REVIEW", "BLOCK"}:
            raise HTTPException(status_code=422, detail="Invalid decision filter")
        try:
            if (
                from_time is not None
                and (from_time.tzinfo is None or from_time.utcoffset() is None)
            ) or (to_time is not None and (to_time.tzinfo is None or to_time.utcoffset() is None)):
                raise ValueError("Prediction time filters must be timezone-aware")
            if from_time is not None and to_time is not None and from_time >= to_time:
                raise ValueError("from_time must precede to_time")
            items = await asyncio.to_thread(
                request.app.state.service.list_predictions,
                limit=limit,
                offset=offset,
                from_time=from_time,
                to_time=to_time,
                decision=decision,
            )
            return {"items": items, "count": len(items), "limit": limit, "offset": offset}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=503, detail="Durable prediction store is unavailable"
            ) from error

    @application.get(
        "/predictions/{transaction_id}",
        response_model=StoredPredictionResult,
        response_model_exclude_none=True,
        description="Look up a stored prediction and safe audit identifiers, without raw payload.",
    )
    async def stored_prediction(request: Request, transaction_id: int) -> StoredPredictionResult:
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="Durable inference store is not ready")
        try:
            found = await asyncio.to_thread(
                request.app.state.service.lookup_prediction, transaction_id
            )
        except Exception as error:
            raise HTTPException(
                status_code=503, detail="Durable inference store is unavailable"
            ) from error
        if found is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return StoredPredictionResult.model_validate(found)

    def feedback_service(request: Request) -> FeedbackService:
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="Durable feedback store is not ready")
        return request.app.state.feedback

    async def ingest_labels(request: Request, labels: list[LabelInput]) -> list[LabelResult]:
        try:
            rows = await asyncio.to_thread(feedback_service(request).ingest, labels)
        except UnknownPrediction as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except LabelConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail="Feedback store is unavailable") from error
        return [LabelResult.model_validate(row) for row in rows]

    @application.post("/labels", response_model=LabelResult)
    async def post_label(request: Request, body: LabelInput) -> LabelResult:
        return (await ingest_labels(request, [body]))[0]

    @application.post("/labels/batch", response_model=LabelBatchResult)
    async def post_labels_batch(request: Request, body: LabelBatchInput) -> LabelBatchResult:
        labels = await ingest_labels(request, body.labels)
        return LabelBatchResult(
            labels=labels,
            count=len(labels),
            created_count=sum(not label.replayed for label in labels),
        )

    @application.get("/labels/{transaction_id}", response_model=LabelResult)
    async def get_label(request: Request, transaction_id: int) -> LabelResult:
        try:
            row = await asyncio.to_thread(feedback_service(request).lookup, transaction_id)
        except Exception as error:
            raise HTTPException(status_code=503, detail="Feedback store is unavailable") from error
        if row is None:
            raise HTTPException(status_code=404, detail="Label not found")
        return LabelResult.model_validate(row)

    @application.get("/feedback/summary")
    async def feedback_summary(
        request: Request,
        evaluation_time: datetime | None = None,
        bundle_version: str | None = None,
        rules_enabled: bool | None = None,
        from_time: datetime | None = FROM_QUERY,
        to_time: datetime | None = TO_QUERY,
    ) -> dict:
        try:
            at = utc(evaluation_time) if evaluation_time is not None else datetime.now(UTC)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            return await asyncio.to_thread(
                feedback_service(request).summary,
                at,
                bundle_version,
                rules_enabled,
                from_time,
                to_time,
            )
        except Exception as error:
            raise HTTPException(status_code=503, detail="Feedback store is unavailable") from error

    def monitoring_service(request: Request) -> MonitoringService:
        if not request.app.state.ready or not hasattr(request.app.state, "monitoring"):
            raise HTTPException(status_code=503, detail="Monitoring service is not ready")
        return request.app.state.monitoring

    async def monitoring_evaluation(
        request: Request,
        window: str,
        from_time: datetime | None,
        to_time: datetime | None,
        persist: bool = False,
    ) -> dict:
        evaluated_at = datetime.now(UTC)
        try:
            start, end = choose_window(window, evaluated_at, from_time, to_time)
            return await asyncio.to_thread(
                monitoring_service(request).evaluate,
                start,
                end,
                evaluated_at,
                database_ready=request.app.state.database_ready,
                state_recovered=request.app.state.state_recovered,
                persist=persist,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=503, detail="Monitoring store is unavailable"
            ) from error

    @application.get("/monitoring/summary")
    async def monitoring_summary(
        request: Request,
        window: str = "24h",
        from_time: datetime | None = FROM_QUERY,
        to_time: datetime | None = TO_QUERY,
    ) -> dict:
        return await monitoring_evaluation(request, window, from_time, to_time)

    @application.get("/monitoring/drift")
    async def monitoring_drift(
        request: Request,
        window: str = "24h",
        from_time: datetime | None = FROM_QUERY,
        to_time: datetime | None = TO_QUERY,
    ) -> dict:
        result = await monitoring_evaluation(request, window, from_time, to_time)
        return {
            key: result[key]
            for key in ("window_start", "window_end", "evaluated_at", "drift", "alerts")
        }

    @application.get("/monitoring/alerts")
    async def monitoring_alerts(
        request: Request, limit: int = 100, severity: str | None = None
    ) -> dict:
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
        if severity is not None and severity.upper() not in {"INFO", "WARNING", "CRITICAL"}:
            raise HTTPException(status_code=422, detail="Invalid alert severity")
        try:
            alerts = await asyncio.to_thread(
                monitoring_service(request).list_alerts, limit, severity
            )
            return {"alerts": alerts, "count": len(alerts)}
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=503, detail="Monitoring store is unavailable"
            ) from error

    @application.post("/monitoring/snapshot")
    async def monitoring_snapshot(
        request: Request,
        window: str = "24h",
        from_time: datetime | None = FROM_QUERY,
        to_time: datetime | None = TO_QUERY,
    ) -> dict:
        return await monitoring_evaluation(request, window, from_time, to_time, persist=True)

    return application


app = create_app()
