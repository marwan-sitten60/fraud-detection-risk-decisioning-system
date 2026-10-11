"""Pure helpers for component TreeSHAP diagnostics."""

import numpy as np
import pandas as pd

HISTORICAL_AMOUNT = frozenset(
    {
        "card1_previous_avg_amount",
        "card1_previous_max_amount",
        "amount_vs_card1_previous_avg",
        "addr1_previous_avg_amount",
        "amount_vs_addr1_previous_avg",
    }
)
COMPOSITE_PREFIXES = (
    "card1_card2_addr1_",
    "card1_addr1_p_emaildomain_",
    "card1_card2_addr1_productcd_",
    "card1_addr1_deviceinfo_",
)
FAMILIES = (
    "raw_transaction",
    "raw_identity",
    "frequency_encoding",
    "historical_amount",
    "composite_identity",
    "composite_temporal_te",
)


def feature_family(name: str, identity_columns=()):
    """Map the frozen feature names to six disjoint, exhaustive source families."""
    if name.endswith("_target_encoded_fraud_rate"):
        return "composite_temporal_te"
    if name in HISTORICAL_AMOUNT:
        return "historical_amount"
    if name.startswith(COMPOSITE_PREFIXES):
        return "composite_identity"
    if name.endswith("_frequency"):
        return "frequency_encoding"
    if (
        name in set(identity_columns)
        or name.startswith("id_")
        or name in ("DeviceType", "DeviceInfo")
    ):
        return "raw_identity"
    return "raw_transaction"


def aggregate_shap(values, feature_names, identity_columns=()):
    """Return per-feature and per-family mean absolute raw-margin SHAP values."""
    matrix = np.asarray(values, dtype=float)
    names = tuple(feature_names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or not len(names):
        raise ValueError("SHAP matrix and feature names must align")
    if len(set(names)) != len(names) or not np.isfinite(matrix).all():
        raise ValueError("SHAP names must be unique and values finite")
    feature = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(matrix).mean(axis=0)})
    feature["family"] = [feature_family(name, identity_columns) for name in names]
    feature = feature.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    feature["rank"] = np.arange(1, len(feature) + 1)
    family = feature.groupby("family", as_index=False).mean_abs_shap.sum()
    family["share"] = family.mean_abs_shap / family.mean_abs_shap.sum()
    family = family.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    family["rank"] = np.arange(1, len(family) + 1)
    return feature, family


def top_local_drivers(shap_row, feature_names, *, count=5):
    values = np.asarray(shap_row, dtype=float)
    if values.ndim != 1 or len(values) != len(feature_names):
        raise ValueError("Local SHAP values must align with feature names")
    order = np.argsort(-np.abs(values), kind="mergesort")[:count]
    return [(feature_names[i], float(values[i])) for i in order]
