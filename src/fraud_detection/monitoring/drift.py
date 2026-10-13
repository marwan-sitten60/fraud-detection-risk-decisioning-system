"""Stable PSI and Jensen-Shannon comparisons against fixed reference bins."""

import numpy as np
import pandas as pd

EPSILON = 1e-6


def numeric_psi(values, edge_values: list, reference: list[float]) -> dict:
    """PSI over reference edges plus a separate missing bucket."""
    n = len(values)
    if n == 0:
        return {"sample_count": 0, "psi": None, "frequencies": None}
    edges = np.asarray(
        [-np.inf if x == "-inf" else np.inf if x == "inf" else float(x) for x in edge_values],
        dtype=float,
    )
    series = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(series)
    bins = np.searchsorted(edges[1:-1], series[~missing], side="left")
    counts = np.bincount(bins, minlength=len(edges) - 1).astype(float)
    counts = np.append(counts, float(missing.sum()))
    actual = counts / counts.sum()
    expected = np.asarray(reference, dtype=float)
    expected = expected / expected.sum()
    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    expected /= expected.sum()
    actual /= actual.sum()
    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    return {"sample_count": n, "psi": psi, "frequencies": actual.tolist()}


def categorical_js(values, reference: dict) -> dict:
    """Jensen-Shannon divergence; unknown categories occupy __UNSEEN__."""
    series = pd.Series(values, dtype="object")
    if series.empty:
        return {"sample_count": 0, "js_divergence": None, "unseen_rate": None, "frequencies": None}
    ref = reference["frequencies"]
    keys = sorted(set(ref) | {"__UNSEEN__", "__MISSING__"})
    known = set(reference["known_categories"])
    mapped = []
    unseen = 0
    for value in series:
        if pd.isna(value):
            mapped.append("__MISSING__")
        elif str(value) in known:
            mapped.append(str(value))
        elif "__OTHER__" in ref and str(value) in reference.get("all_known_categories", []):
            mapped.append("__OTHER__")
        else:
            mapped.append("__UNSEEN__")
            unseen += 1
    actual_counts = pd.Series(mapped).value_counts().reindex(keys, fill_value=0).to_numpy(float)
    actual = actual_counts / actual_counts.sum()
    expected = np.asarray([ref.get(key, 0.0) for key in keys], dtype=float)
    expected /= expected.sum()
    actual = np.clip(actual, EPSILON, None)
    expected = np.clip(expected, EPSILON, None)
    actual /= actual.sum()
    expected /= expected.sum()
    middle = (actual + expected) / 2
    js = float(
        0.5 * np.sum(actual * np.log(actual / middle))
        + 0.5 * np.sum(expected * np.log(expected / middle))
    )
    return {
        "sample_count": len(series),
        "js_divergence": js,
        "unseen_rate": unseen / len(series),
        "frequencies": {key: float(value) for key, value in zip(keys, actual, strict=True)},
    }
