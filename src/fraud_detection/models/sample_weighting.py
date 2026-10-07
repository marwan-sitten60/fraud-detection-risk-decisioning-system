"""Deterministic sample weighting for chronological model fitting."""

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400.0


def exponential_time_decay_weights(
    fit_transaction_dt: pd.Series | np.ndarray,
    *,
    half_life_days: float,
) -> np.ndarray:
    """Return mean-one exponential weights derived only from fit timestamps.

    The latest timestamp supplied by the caller is the reference time. A row one
    half-life older receives half its raw weight. Normalization divides all raw
    weights by their mean and therefore preserves every relative weight ratio.
    """
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be a positive finite number")

    timestamps = pd.to_numeric(pd.Series(fit_transaction_dt, copy=True), errors="raise")
    if timestamps.empty:
        raise ValueError("Fit timestamps must not be empty")
    timestamp_values = timestamps.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(timestamp_values).all():
        raise ValueError("Fit timestamps must contain finite, non-missing values")

    reference_time = float(timestamp_values.max())
    age_days = (reference_time - timestamp_values) / SECONDS_PER_DAY
    raw_weights = np.exp(-np.log(2.0) * age_days / float(half_life_days))
    mean_weight = float(raw_weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Time-decay weights could not be normalized")
    return raw_weights / mean_weight
