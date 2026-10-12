"""Explicit request and response models for frozen risk scoring."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["APPROVE", "REVIEW", "BLOCK"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction: dict[str, Any]
    include_rules: bool | None = None


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[dict[str, Any]]
    include_rules: bool | None = None


class PredictionResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    TransactionID: int
    lightgbm_score: float = Field(ge=0, le=1)
    xgboost_score: float = Field(ge=0, le=1)
    ensemble_score: float = Field(ge=0, le=1)
    calibrated_probability: float = Field(ge=0, le=1)
    base_decision: Decision
    final_decision: Decision | None = None
    triggered_rule_ids: list[str] | None = None
    expected_cost_approve: float = Field(ge=0)
    expected_cost_review: float = Field(ge=0)
    expected_cost_block: float = Field(ge=0)
    bundle_version: str


class BatchPredictionResult(BaseModel):
    predictions: list[PredictionResult]
    count: int
    bundle_version: str


class ReadinessResult(BaseModel):
    ready: bool
    bundle_version: str
    feature_count: int
    rules_enabled: bool
    database_ready: bool
    state_recovered: bool
    replayed_transaction_count: int


class StoredPredictionResult(PredictionResult):
    request_id: str
    scored_at: datetime
    rules_enabled: bool


class VersionResult(BaseModel):
    service_version: str
    bundle_version: str
    feature_count: int
    ensemble: str
    calibration_method: str
    primary_policy: str
    rules_enabled: bool
