"""Stable, label-free JSON for idempotency and deterministic replay."""

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

FORBIDDEN = frozenset({"isFraud", "y_true"})


def normalize(value: Any) -> Any:
    """Convert numpy scalars and missing values without changing finite values."""
    if isinstance(value, np.generic):
        return normalize(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        if FORBIDDEN.intersection(value):
            raise ValueError("Target fields are forbidden")
        return {str(key): normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    raise ValueError("Unsupported raw transaction scalar")


def canonical_json(value: Any) -> str:
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scoring_payload(endpoint: str, rows: list[dict], rules_enabled: bool) -> dict:
    return {"endpoint": endpoint, "rules_enabled": rules_enabled, "transactions": rows}
