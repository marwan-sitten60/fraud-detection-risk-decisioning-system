"""Reusable LightGBM utilities for leakage-safe temporal model selection."""

import gc
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from fraud_detection.features import (
    BehavioralFeatureEngineer,
    CompositeFeatureEngineer,
    GraphFeatureEngineer,
    LabelAvailabilityStatistics,
    Phase1FeatureEngineer,
    TemporalTargetEncoder,
)
from fraud_detection.models.comparison import TrainingOnlyPreprocessor


@dataclass(frozen=True, slots=True)
class TemporalCVFold:
    """Positional indices for one expanding chronological fold."""

    train_indices: np.ndarray
    evaluation_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Ranking metrics and effective iteration for one fitted fold."""

    pr_auc: float
    roc_auc: float
    best_iteration: int


@dataclass(frozen=True, slots=True)
class LightGBMCVResult:
    """Aggregated chronological CV evidence for one parameter configuration."""

    parameters: dict[str, Any]
    fold_pr_auc: tuple[float, ...]
    fold_roc_auc: tuple[float, ...]
    best_iterations: tuple[int, ...]
    mean_pr_auc: float
    std_pr_auc: float
    mean_roc_auc: float


@dataclass(frozen=True, slots=True)
class PreparedMatrices:
    """Train-fitted numeric matrices for one chronological fit/evaluation pair."""

    x_train: np.ndarray
    x_evaluation: np.ndarray
    feature_count: int
    feature_names: tuple[str, ...] = ()
    target_encoding_train_statistics: LabelAvailabilityStatistics | None = None
    target_encoding_evaluation_statistics: LabelAvailabilityStatistics | None = None


def make_expanding_time_folds(
    dataframe: pd.DataFrame,
    *,
    n_splits: int = 3,
    initial_train_fraction: float = 0.70,
) -> list[TemporalCVFold]:
    """Build expanding folds with boundaries aligned to complete timestamps."""
    if dataframe.empty:
        raise ValueError("Temporal CV input must not be empty")
    if "TransactionDT" not in dataframe.columns:
        raise ValueError("Temporal CV input requires TransactionDT")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be between 0 and 1")

    times = pd.to_numeric(dataframe["TransactionDT"], errors="raise").to_numpy()
    if pd.isna(times).any():
        raise ValueError("TransactionDT must not contain missing values")
    if np.any(times[1:] < times[:-1]):
        raise ValueError("Temporal CV input must be sorted chronologically")

    row_count = len(dataframe)
    initial_position = int(row_count * initial_train_fraction)
    if initial_position <= 0 or initial_position >= row_count:
        raise ValueError("initial_train_fraction produces an empty temporal period")

    desired_boundaries = np.linspace(initial_position, row_count, n_splits + 1)
    boundaries = [
        int(np.searchsorted(times, times[min(int(position), row_count - 1)], side="left"))
        for position in desired_boundaries[:-1]
    ]
    boundaries.append(row_count)
    if boundaries[0] == 0 or any(
        right <= left for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        raise ValueError("Temporal CV configuration produces an empty fold")

    folds: list[TemporalCVFold] = []
    for fold_index in range(n_splits):
        train_end = boundaries[fold_index]
        evaluation_end = boundaries[fold_index + 1]
        train_indices = np.arange(train_end, dtype=np.int64)
        evaluation_indices = np.arange(train_end, evaluation_end, dtype=np.int64)
        if times[train_indices[-1]] >= times[evaluation_indices[0]]:
            raise ValueError("Temporal fold boundaries must separate complete timestamps")
        folds.append(TemporalCVFold(train_indices, evaluation_indices))
    return folds


def evaluate_lightgbm_fold(
    parameters: dict[str, Any],
    x_train: np.ndarray,
    y_train: pd.Series | np.ndarray,
    x_evaluation: np.ndarray,
    y_evaluation: pd.Series | np.ndarray,
    *,
    early_stopping_rounds: int = 50,
) -> FoldEvaluation:
    """Fit one fold using only its later chronological partition for early stopping."""
    if early_stopping_rounds <= 0:
        raise ValueError("early_stopping_rounds must be positive")
    model = lgb.LGBMClassifier(**parameters)
    model.fit(
        x_train,
        y_train,
        eval_X=x_evaluation,
        eval_y=y_evaluation,
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(
                early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )
    best_iteration = int(model.best_iteration_ or parameters.get("n_estimators", 100))
    probabilities = model.predict_proba(x_evaluation, num_iteration=best_iteration)[:, 1]
    return FoldEvaluation(
        pr_auc=float(average_precision_score(y_evaluation, probabilities)),
        roc_auc=float(roc_auc_score(y_evaluation, probabilities)),
        best_iteration=best_iteration,
    )


def summarize_lightgbm_cv(
    parameters: dict[str, Any], fold_evaluations: list[FoldEvaluation]
) -> LightGBMCVResult:
    """Aggregate fold-level metrics for configuration selection by mean PR-AUC."""
    if not fold_evaluations:
        raise ValueError("At least one fold evaluation is required")
    pr_auc = np.asarray([evaluation.pr_auc for evaluation in fold_evaluations])
    roc_auc = np.asarray([evaluation.roc_auc for evaluation in fold_evaluations])
    iterations = tuple(evaluation.best_iteration for evaluation in fold_evaluations)
    return LightGBMCVResult(
        parameters=dict(parameters),
        fold_pr_auc=tuple(float(value) for value in pr_auc),
        fold_roc_auc=tuple(float(value) for value in roc_auc),
        best_iterations=iterations,
        mean_pr_auc=float(pr_auc.mean()),
        std_pr_auc=float(pr_auc.std(ddof=0)),
        mean_roc_auc=float(roc_auc.mean()),
    )


def prepare_engineered_matrices(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    include_graph: bool,
    include_composite: bool = False,
    include_target_encoding: bool = False,
    target_encoding_label_delay_days: float = 0.0,
    scaling_batch_rows: int = 50_000,
) -> PreparedMatrices:
    """Fit the selected existing feature pipeline on one historical fold only."""
    if scaling_batch_rows <= 0:
        raise ValueError("scaling_batch_rows must be positive")
    predictor_columns = [
        column for column in train.columns if column not in ("isFraud", "TransactionID")
    ]
    train_predictors = train.loc[:, predictor_columns]
    evaluation_predictors = evaluation.loc[:, predictor_columns]

    frequency_engineer = Phase1FeatureEngineer(
        include_amount_time=False,
        include_missing_indicators=False,
        include_frequency_encodings=True,
    ).fit(train_predictors)
    frequency_train = frequency_engineer.transform(train_predictors)
    frequency_evaluation = frequency_engineer.transform(evaluation_predictors)

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
    engineered_train = pd.concat([frequency_train, amount_train], axis=1)
    engineered_evaluation = pd.concat([frequency_evaluation, amount_evaluation], axis=1)

    if include_composite:
        composite_engineer = CompositeFeatureEngineer()
        composite_train = composite_engineer.fit_transform_features(train)
        composite_evaluation = composite_engineer.transform_features(evaluation, update_state=True)
        engineered_train = pd.concat([engineered_train, composite_train], axis=1)
        engineered_evaluation = pd.concat([engineered_evaluation, composite_evaluation], axis=1)

    if include_target_encoding:
        target_encoder = TemporalTargetEncoder(
            alpha=50.0,
            initial_prior=0.5,
            label_delay_days=target_encoding_label_delay_days,
        )
        target_encoded_train = target_encoder.fit_transform_features(train)
        target_encoded_evaluation = target_encoder.transform_features(evaluation)
        engineered_train = pd.concat([engineered_train, target_encoded_train], axis=1)
        engineered_evaluation = pd.concat(
            [engineered_evaluation, target_encoded_evaluation], axis=1
        )
        target_encoding_train_statistics = target_encoder.training_statistics_
        target_encoding_evaluation_statistics = target_encoder.last_transform_statistics_
    else:
        target_encoding_train_statistics = None
        target_encoding_evaluation_statistics = None

    if include_graph:
        graph_columns = ["TransactionID", "TransactionDT", "card1", "DeviceInfo", "addr1"]
        graph_engineer = GraphFeatureEngineer()
        graph_train = graph_engineer.fit_transform_features(train.loc[:, graph_columns])
        graph_evaluation = graph_engineer.transform_features(evaluation.loc[:, graph_columns])
        engineered_train = pd.concat([engineered_train, graph_train], axis=1)
        engineered_evaluation = pd.concat([engineered_evaluation, graph_evaluation], axis=1)

    preprocessor = TrainingOnlyPreprocessor()
    x_train = preprocessor.fit_transform(engineered_train)
    x_evaluation = preprocessor.transform(engineered_evaluation)
    feature_count = x_train.shape[1]
    feature_names = tuple(preprocessor.feature_names_)

    del (
        train_predictors,
        evaluation_predictors,
        frequency_train,
        frequency_evaluation,
        amount_train,
        amount_evaluation,
        engineered_train,
        engineered_evaluation,
        frequency_engineer,
        amount_engineer,
        preprocessor,
    )
    if include_graph:
        del graph_train, graph_evaluation, graph_engineer
    if include_composite:
        del composite_train, composite_evaluation, composite_engineer
    if include_target_encoding:
        del target_encoded_train, target_encoded_evaluation, target_encoder
    gc.collect()

    scaler = StandardScaler(copy=False)
    for start in range(0, len(x_train), scaling_batch_rows):
        scaler.partial_fit(x_train[start : start + scaling_batch_rows])
    x_train = scaler.transform(x_train)
    x_evaluation = scaler.transform(x_evaluation)
    return PreparedMatrices(
        x_train,
        x_evaluation,
        feature_count,
        feature_names,
        target_encoding_train_statistics,
        target_encoding_evaluation_statistics,
    )
