"""Strict, audit-safe label API contracts."""

import json
import math
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(UTC)


class LabelInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: int = Field(alias="TransactionID", gt=0, strict=True)
    is_fraud: StrictBool
    label_source: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    matured_at: datetime
    metadata: dict[str, str | int | float | bool | None] | None = None

    @field_validator("label_source")
    @classmethod
    def source(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", value):
            raise ValueError("label_source must be an audit-safe identifier")
        return value

    @field_validator("observed_at", "matured_at", mode="before")
    @classmethod
    def timestamp_input(cls, value: Any) -> Any:
        if not isinstance(value, (str, datetime)):
            raise ValueError("Timestamp must be an ISO datetime with timezone")
        return value

    @field_validator("observed_at", "matured_at")
    @classmethod
    def timestamp(cls, value: datetime) -> datetime:
        return utc(value)

    @field_validator("metadata")
    @classmethod
    def safe_metadata(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if len(value) > 16 or len(json.dumps(value, allow_nan=False)) > 2048:
            raise ValueError("metadata exceeds safe size")
        for key, item in value.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key):
                raise ValueError("metadata key is invalid")
            if any(
                word in key.lower() for word in ("secret", "password", "token", "payload", "raw")
            ):
                raise ValueError("metadata key is not audit-safe")
            if isinstance(item, str) and len(item) > 256:
                raise ValueError("metadata value is too long")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("metadata number must be finite")
        return value

    @model_validator(mode="after")
    def chronology(self) -> "LabelInput":
        if self.matured_at < self.observed_at:
            raise ValueError("matured_at must be at or after observed_at")
        if self.observed_at > datetime.now(UTC):
            raise ValueError("observed_at cannot be in the future")
        return self


class LabelBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[LabelInput] = Field(min_length=1, max_length=256)


class LabelResult(BaseModel):
    transaction_id: int = Field(alias="TransactionID")
    is_fraud: bool
    label_source: str
    observed_at: datetime
    matured_at: datetime | None
    ingested_at: datetime
    maturity_status: str
    replayed: bool


class LabelBatchResult(BaseModel):
    labels: list[LabelResult]
    count: int
    created_count: int
