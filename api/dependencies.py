"""Operational settings and frozen runtime access for the API."""

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from fraud_detection.inference import FrozenFraudInferenceRuntime

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Serving settings; none can alter fitted model or policy parameters."""

    bundle_path: Path
    database_url: str | None = None
    enable_rules: bool = False
    max_batch_size: int = 256

    @classmethod
    def from_env(cls) -> "ApiSettings":
        raw_path = Path(os.getenv("FRAUD_BUNDLE_PATH", "models/frozen_v1"))
        bundle_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        raw_rules = os.getenv("FRAUD_ENABLE_RULES", "false").strip().lower()
        if raw_rules not in {"true", "false"}:
            raise ValueError("FRAUD_ENABLE_RULES must be true or false")
        maximum = int(os.getenv("FRAUD_MAX_BATCH_SIZE", "256"))
        if maximum <= 0:
            raise ValueError("FRAUD_MAX_BATCH_SIZE must be positive")
        return cls(
            bundle_path=bundle_path,
            database_url=os.getenv("FRAUD_DATABASE_URL"),
            enable_rules=raw_rules == "true",
            max_batch_size=maximum,
        )


def runtime_from_request(request: Request) -> FrozenFraudInferenceRuntime | None:
    return getattr(request.app.state, "runtime", None)
