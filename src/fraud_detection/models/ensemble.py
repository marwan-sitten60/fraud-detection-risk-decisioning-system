"""Deterministic utilities for evaluating simple blends of temporal OOF scores."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OOF_KEYS = ("TransactionID", "fold", "y_true")
MODEL_OOF_COLUMNS = (*OOF_KEYS, "score")
COMBINED_OOF_COLUMNS = (*OOF_KEYS, "lightgbm_score", "xgboost_score")


@dataclass(frozen=True, slots=True)
class DisagreementSummary:
    """Compact comparison of model rankings and missed-fraud scores."""

    material_rank_difference_threshold: float
    materially_different_rank_fraction: float
    lightgbm_missed_fraud_count: int
    mean_xgboost_minus_lightgbm_on_missed_fraud: float
    xgboost_higher_on_missed_fraud_fraction: float
    xgboost_rescue_at_0_50_fraction: float


def _validate_model_oof(frame: pd.DataFrame, model_name: str) -> None:
    missing = [column for column in MODEL_OOF_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{model_name} OOF predictions are missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{model_name} OOF predictions must not be empty")
    if frame.loc[:, list(MODEL_OOF_COLUMNS)].isna().any().any():
        raise ValueError(f"{model_name} OOF predictions contain missing values")
    if frame.duplicated(list(OOF_KEYS)).any():
        raise ValueError(f"{model_name} OOF predictions contain duplicate rows")
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name} OOF targets must contain only 0 and 1")
    scores = frame["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{model_name} OOF scores must be finite probabilities in [0, 1]")


def aggregate_oof_predictions(
    lightgbm_predictions: pd.DataFrame,
    xgboost_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Align two model OOF tables exactly by row identifier, fold, and target."""
    _validate_model_oof(lightgbm_predictions, "LightGBM")
    _validate_model_oof(xgboost_predictions, "XGBoost")

    lightgbm = lightgbm_predictions.loc[:, list(MODEL_OOF_COLUMNS)].rename(
        columns={"score": "lightgbm_score"}
    )
    xgboost = xgboost_predictions.loc[:, list(MODEL_OOF_COLUMNS)].rename(
        columns={"score": "xgboost_score"}
    )
    combined = lightgbm.merge(xgboost, on=list(OOF_KEYS), how="inner", validate="one_to_one")
    if len(combined) != len(lightgbm) or len(combined) != len(xgboost):
        raise ValueError("LightGBM and XGBoost OOF rows are not aligned exactly")
    return combined.loc[:, list(COMBINED_OOF_COLUMNS)].sort_values(
        ["fold", "TransactionID"], kind="mergesort", ignore_index=True
    )


def weighted_score_blend(
    lightgbm_scores: Iterable[float],
    xgboost_scores: Iterable[float],
    *,
    lightgbm_weight: float,
    xgboost_weight: float,
) -> np.ndarray:
    """Return a convex weighted average of two equally aligned score vectors."""
    weights = np.asarray([lightgbm_weight, xgboost_weight], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Blend weights must be finite and non-negative")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Blend weights must sum to 1.0")

    lightgbm = np.asarray(lightgbm_scores, dtype=float)
    xgboost = np.asarray(xgboost_scores, dtype=float)
    if lightgbm.ndim != 1 or xgboost.ndim != 1 or len(lightgbm) != len(xgboost):
        raise ValueError("Model score vectors must be one-dimensional and equally sized")
    if not np.isfinite(lightgbm).all() or not np.isfinite(xgboost).all():
        raise ValueError("Model score vectors must not contain missing or non-finite values")
    return lightgbm_weight * lightgbm + xgboost_weight * xgboost


def prediction_correlations(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson and Spearman score correlations per fold and overall."""
    _validate_combined_oof(oof_predictions)
    rows: list[dict[str, float | int | str]] = []
    groups: list[tuple[str | int, pd.DataFrame]] = [
        (int(fold), frame) for fold, frame in oof_predictions.groupby("fold", sort=True)
    ]
    groups.append(("overall", oof_predictions))
    for fold, frame in groups:
        rows.append(
            {
                "fold": fold,
                "row_count": len(frame),
                "pearson": float(frame["lightgbm_score"].corr(frame["xgboost_score"])),
                "spearman": float(
                    frame["lightgbm_score"].corr(frame["xgboost_score"], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_probability_blends(
    oof_predictions: pd.DataFrame,
    lightgbm_weights: Iterable[float],
) -> pd.DataFrame:
    """Evaluate fixed probability blends using only supplied temporal OOF predictions."""
    _validate_combined_oof(oof_predictions)
    rows: list[dict[str, object]] = []
    for lightgbm_weight in lightgbm_weights:
        xgboost_weight = 1.0 - float(lightgbm_weight)
        scores = weighted_score_blend(
            oof_predictions["lightgbm_score"],
            oof_predictions["xgboost_score"],
            lightgbm_weight=float(lightgbm_weight),
            xgboost_weight=xgboost_weight,
        )
        fold_pr_auc: list[float] = []
        fold_roc_auc: list[float] = []
        for _, indices in oof_predictions.groupby("fold", sort=True).groups.items():
            positions = oof_predictions.index.get_indexer(indices)
            targets = oof_predictions.loc[indices, "y_true"]
            fold_pr_auc.append(float(average_precision_score(targets, scores[positions])))
            fold_roc_auc.append(float(roc_auc_score(targets, scores[positions])))
        rows.append(
            {
                "lightgbm_weight": float(lightgbm_weight),
                "xgboost_weight": xgboost_weight,
                "fold_pr_auc": tuple(fold_pr_auc),
                "fold_roc_auc": tuple(fold_roc_auc),
                "mean_pr_auc": float(np.mean(fold_pr_auc)),
                "std_pr_auc": float(np.std(fold_pr_auc, ddof=0)),
                "mean_roc_auc": float(np.mean(fold_roc_auc)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_pr_auc", "lightgbm_weight"], ascending=[False, False], ignore_index=True
    )


def summarize_disagreement(
    oof_predictions: pd.DataFrame,
    *,
    material_rank_difference: float = 0.20,
) -> DisagreementSummary:
    """Quantify rank disagreement and XGBoost scores on fraud missed at 0.50 by LightGBM."""
    _validate_combined_oof(oof_predictions)
    if not 0 < material_rank_difference <= 1:
        raise ValueError("material_rank_difference must be in (0, 1]")
    lightgbm_rank = oof_predictions["lightgbm_score"].rank(pct=True, method="average")
    xgboost_rank = oof_predictions["xgboost_score"].rank(pct=True, method="average")
    material_fraction = float(
        ((lightgbm_rank - xgboost_rank).abs() >= material_rank_difference).mean()
    )
    missed = oof_predictions.loc[
        (oof_predictions["y_true"] == 1) & (oof_predictions["lightgbm_score"] < 0.50)
    ]
    if missed.empty:
        mean_delta = higher_fraction = rescue_fraction = 0.0
    else:
        delta = missed["xgboost_score"] - missed["lightgbm_score"]
        mean_delta = float(delta.mean())
        higher_fraction = float((delta > 0).mean())
        rescue_fraction = float((missed["xgboost_score"] >= 0.50).mean())
    return DisagreementSummary(
        material_rank_difference_threshold=material_rank_difference,
        materially_different_rank_fraction=material_fraction,
        lightgbm_missed_fraud_count=len(missed),
        mean_xgboost_minus_lightgbm_on_missed_fraud=mean_delta,
        xgboost_higher_on_missed_fraud_fraction=higher_fraction,
        xgboost_rescue_at_0_50_fraction=rescue_fraction,
    )


def _validate_combined_oof(frame: pd.DataFrame) -> None:
    missing = [column for column in COMBINED_OOF_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Combined OOF predictions are missing columns: {missing}")
    if frame.empty:
        raise ValueError("Combined OOF predictions must not be empty")
    if frame.loc[:, list(COMBINED_OOF_COLUMNS)].isna().any().any():
        raise ValueError("Combined OOF predictions contain missing values")
    if frame.duplicated(list(OOF_KEYS)).any():
        raise ValueError("Combined OOF predictions contain duplicate rows")
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise ValueError("Combined OOF targets must contain only 0 and 1")
    for column in ("lightgbm_score", "xgboost_score"):
        scores = frame[column].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"{column} must contain finite probabilities in [0, 1]")
