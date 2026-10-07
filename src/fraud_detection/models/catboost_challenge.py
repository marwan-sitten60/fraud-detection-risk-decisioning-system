"""Native-categorical CatBoost preparation and temporal evaluation utilities."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.features import BehavioralFeatureEngineer

MISSING_CATEGORY = "__MISSING__"
ENTITY_CATEGORICAL_COLUMNS = (
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
)


@dataclass(frozen=True, slots=True)
class PreparedCatBoostData:
    """Native feature frames and categorical metadata for one temporal period pair."""

    train: pd.DataFrame
    evaluation: pd.DataFrame
    categorical_indices: tuple[int, ...]
    categorical_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return self.train.shape[1]


@dataclass(frozen=True, slots=True)
class CatBoostFoldEvaluation:
    """Ranking metrics and effective tree count for one CatBoost fold."""

    pr_auc: float
    roc_auc: float
    best_iteration: int


@dataclass(frozen=True, slots=True)
class CatBoostCVResult:
    """Aggregated temporal-CV evidence for one CatBoost configuration."""

    parameters: dict[str, Any]
    fold_pr_auc: tuple[float, ...]
    fold_roc_auc: tuple[float, ...]
    best_iterations: tuple[int, ...]
    mean_pr_auc: float
    std_pr_auc: float
    mean_roc_auc: float


def prepare_catboost_data(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> PreparedCatBoostData:
    """Create raw-plus-historical-amount frames without target or ID leakage."""
    excluded = {"isFraud", "TransactionID"}
    predictor_columns = [column for column in train.columns if column not in excluded]
    missing_evaluation = [
        column for column in predictor_columns if column not in evaluation.columns
    ]
    if missing_evaluation:
        raise ValueError(f"Evaluation data is missing predictors: {missing_evaluation}")

    train_features = train.loc[:, predictor_columns].copy()
    evaluation_features = evaluation.loc[:, predictor_columns].copy()
    behavior_columns = [
        column
        for column in (
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
            "addr1",
            "DeviceInfo",
            "P_emaildomain",
        )
        if column in train.columns
    ]
    amount_engineer = BehavioralFeatureEngineer(
        include_counts=False,
        include_amounts=True,
        include_recency=False,
        include_velocity=False,
    )
    amount_train = amount_engineer.fit_transform_features(train.loc[:, behavior_columns])
    amount_evaluation = amount_engineer.transform_features(
        evaluation.loc[:, behavior_columns], update_state=True
    )
    train_features = pd.concat([train_features, amount_train], axis=1)
    evaluation_features = pd.concat([evaluation_features, amount_evaluation], axis=1)

    dtype_categories = set(
        train_features.select_dtypes(include=["object", "string", "category"]).columns
    )
    explicit_categories = {
        column for column in ENTITY_CATEGORICAL_COLUMNS if column in train_features.columns
    }
    categorical_names = tuple(
        column
        for column in train_features.columns
        if column in dtype_categories or column in explicit_categories
    )
    for column in categorical_names:
        train_features[column] = _normalize_category(train_features[column])
        evaluation_features[column] = _normalize_category(evaluation_features[column])

    categorical_indices = tuple(
        train_features.columns.get_loc(column) for column in categorical_names
    )
    return PreparedCatBoostData(
        train=train_features,
        evaluation=evaluation_features,
        categorical_indices=categorical_indices,
        categorical_names=categorical_names,
    )


def evaluate_catboost_fold(
    parameters: dict[str, Any],
    prepared: PreparedCatBoostData,
    y_train: pd.Series | np.ndarray,
    y_evaluation: pd.Series | np.ndarray,
    *,
    early_stopping_rounds: int = 50,
) -> CatBoostFoldEvaluation:
    """Fit one chronological fold with fold-local monitoring and native categories."""
    if early_stopping_rounds <= 0:
        raise ValueError("early_stopping_rounds must be positive")
    train_pool = Pool(prepared.train, y_train, cat_features=prepared.categorical_indices)
    evaluation_pool = Pool(
        prepared.evaluation,
        y_evaluation,
        cat_features=prepared.categorical_indices,
    )
    model = CatBoostClassifier(**parameters)
    model.fit(
        train_pool,
        eval_set=evaluation_pool,
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True,
        verbose=False,
    )
    best_iteration = int(model.get_best_iteration() + 1)
    probabilities = model.predict_proba(evaluation_pool)[:, 1]
    return CatBoostFoldEvaluation(
        pr_auc=float(average_precision_score(y_evaluation, probabilities)),
        roc_auc=float(roc_auc_score(y_evaluation, probabilities)),
        best_iteration=best_iteration,
    )


def summarize_catboost_cv(
    parameters: dict[str, Any],
    fold_evaluations: list[CatBoostFoldEvaluation],
) -> CatBoostCVResult:
    """Aggregate fold evidence for selection by mean Average Precision."""
    if not fold_evaluations:
        raise ValueError("At least one fold evaluation is required")
    pr_auc = np.asarray([evaluation.pr_auc for evaluation in fold_evaluations])
    roc_auc = np.asarray([evaluation.roc_auc for evaluation in fold_evaluations])
    return CatBoostCVResult(
        parameters=dict(parameters),
        fold_pr_auc=tuple(float(value) for value in pr_auc),
        fold_roc_auc=tuple(float(value) for value in roc_auc),
        best_iterations=tuple(evaluation.best_iteration for evaluation in fold_evaluations),
        mean_pr_auc=float(pr_auc.mean()),
        std_pr_auc=float(pr_auc.std(ddof=0)),
        mean_roc_auc=float(roc_auc.mean()),
    )


def _normalize_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_CATEGORY).astype(str)
