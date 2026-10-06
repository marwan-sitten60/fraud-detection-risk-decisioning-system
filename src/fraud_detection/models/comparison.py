"""Leakage-safe baseline model comparison on a fixed temporal validation set."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Validation metrics for fraud probabilities and a 0.50 decision threshold."""

    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


@dataclass(slots=True)
class SearchResult:
    """Best model and full temporal parameter-search history."""

    model_name: str
    best_parameters: dict[str, Any]
    metrics: BinaryMetrics
    model: Any
    validation_probabilities: np.ndarray
    best_run_id: str
    history: pd.DataFrame


class TrainingOnlyPreprocessor:
    """Median-impute numeric fields and frequency-encode categories using train only."""

    def __init__(self) -> None:
        self.feature_names_: list[str] = []
        self.categorical_features_: set[str] = set()
        self.numeric_medians_: dict[str, float] = {}
        self.frequency_maps_: dict[str, dict[str, float]] = {}
        self.is_fitted_ = False

    def fit(self, dataframe: pd.DataFrame) -> "TrainingOnlyPreprocessor":
        """Learn numeric medians and category frequencies from training data."""
        if dataframe.empty:
            raise ValueError("Training predictors must not be empty")

        self.feature_names_ = dataframe.columns.tolist()
        categorical_columns = dataframe.select_dtypes(
            include=["object", "string", "category"]
        ).columns
        self.categorical_features_ = set(categorical_columns)
        for feature in self.feature_names_:
            values = dataframe[feature]
            if feature in self.categorical_features_:
                normalized = values.astype("string").fillna("<MISSING>")
                self.frequency_maps_[feature] = (
                    normalized.value_counts(normalize=True).astype(float).to_dict()
                )
            else:
                numeric_values = pd.to_numeric(values, errors="coerce")
                median = numeric_values.median()
                self.numeric_medians_[feature] = float(median) if pd.notna(median) else 0.0
        self.is_fitted_ = True
        return self

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Apply train-derived preprocessing; unseen categories map to zero frequency."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform")
        missing = [feature for feature in self.feature_names_ if feature not in dataframe.columns]
        if missing:
            raise ValueError(f"Predictor columns are missing during transform: {missing}")

        matrix = np.empty((len(dataframe), len(self.feature_names_)), dtype=np.float32)
        for index, feature in enumerate(self.feature_names_):
            values = dataframe[feature]
            if feature in self.categorical_features_:
                normalized = values.astype("string").fillna("<MISSING>")
                matrix[:, index] = (
                    normalized.map(self.frequency_maps_[feature]).fillna(0.0).to_numpy(np.float32)
                )
            else:
                numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(
                    dtype=np.float32, na_value=np.nan
                )
                invalid_mask = ~np.isfinite(numeric_values)
                numeric_values[invalid_mask] = self.numeric_medians_[feature]
                matrix[:, index] = numeric_values
        return matrix

    def fit_transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Fit on training data and transform it."""
        return self.fit(dataframe).transform(dataframe)


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray
) -> BinaryMetrics:
    """Calculate ranking metrics and threshold metrics at 0.50."""
    predictions = (probabilities >= 0.50).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return BinaryMetrics(
        pr_auc=float(average_precision_score(y_true, probabilities)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
    )


def run_temporal_parameter_search(
    model_name: str,
    model_factory: Callable[[dict[str, Any]], Any],
    parameter_grid: dict[str, list[Any]],
    x_train: np.ndarray,
    y_train: pd.Series | np.ndarray,
    x_validation: np.ndarray,
    y_validation: pd.Series | np.ndarray,
    experiment_name: str = "fraud-detection-model-comparison",
) -> SearchResult:
    """Evaluate each parameter combination on one fixed chronological validation set."""
    mlflow.set_experiment(experiment_name)
    rows: list[dict[str, Any]] = []
    best_model: Any = None
    best_metrics: BinaryMetrics | None = None
    best_parameters: dict[str, Any] = {}
    best_probabilities: np.ndarray | None = None
    best_run_id = ""

    for parameters in ParameterGrid(parameter_grid):
        with mlflow.start_run(run_name=f"{model_name}-temporal-grid") as run:
            model = model_factory(parameters)
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            metrics = evaluate_probabilities(y_validation, probabilities)

            mlflow.set_tags(
                {
                    "stage": "experimentation",
                    "dataset": "ieee-cis",
                    "split": "temporal-validation",
                    "model_name": model_name,
                }
            )
            mlflow.log_params({"model_name": model_name, **parameters})
            mlflow.log_metrics(
                {
                    "pr_auc": metrics.pr_auc,
                    "roc_auc": metrics.roc_auc,
                    "precision_at_0_50": metrics.precision,
                    "recall_at_0_50": metrics.recall,
                    "f1_at_0_50": metrics.f1,
                    "train_rows": len(y_train),
                    "validation_rows": len(y_validation),
                    "feature_count": x_train.shape[1],
                }
            )
            rows.append(
                {
                    "model": model_name,
                    "parameters": parameters,
                    "run_id": run.info.run_id,
                    "pr_auc": metrics.pr_auc,
                    "roc_auc": metrics.roc_auc,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )
            if best_metrics is None or metrics.pr_auc > best_metrics.pr_auc:
                best_model = model
                best_metrics = metrics
                best_parameters = dict(parameters)
                best_probabilities = probabilities
                best_run_id = run.info.run_id

    if best_metrics is None or best_probabilities is None:
        raise RuntimeError(f"No parameter configurations were evaluated for {model_name}")

    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("best_for_model", "true")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    history = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    LOGGER.info(
        "%s best temporal validation PR-AUC %.6f with %s",
        model_name,
        best_metrics.pr_auc,
        best_parameters,
    )
    return SearchResult(
        model_name=model_name,
        best_parameters=best_parameters,
        metrics=best_metrics,
        model=best_model,
        validation_probabilities=best_probabilities,
        best_run_id=best_run_id,
        history=history,
    )


def save_model_comparison_plots(
    results: list[SearchResult],
    y_validation: pd.Series | np.ndarray,
    figures_dir: Path,
) -> list[Path]:
    """Save combined PR/ROC curves and the validation winner's confusion matrix."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    pr_path = figures_dir / "validation_precision_recall_comparison.png"
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    for result in results:
        precision, recall, _ = precision_recall_curve(y_validation, result.validation_probabilities)
        axis.plot(recall, precision, label=f"{result.model_name} ({result.metrics.pr_auc:.3f})")
    axis.axhline(float(np.mean(y_validation)), linestyle="--", color="gray", label="No skill")
    axis.set(title="Temporal validation precision-recall", xlabel="Recall", ylabel="Precision")
    axis.legend()
    fig.tight_layout()
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)
    paths.append(pr_path)

    roc_path = figures_dir / "validation_roc_comparison.png"
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    for result in results:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_validation, result.validation_probabilities
        )
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{result.model_name} ({result.metrics.roc_auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(
        title="Temporal validation ROC", xlabel="False positive rate", ylabel="True positive rate"
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)
    paths.append(roc_path)

    winner = max(results, key=lambda result: result.metrics.pr_auc)
    confusion_path = figures_dir / "validation_winner_confusion_matrix.png"
    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.asarray(winner.metrics.confusion_matrix)).plot(
        ax=axis, colorbar=False, cmap="Blues"
    )
    axis.set_title(f"{winner.model_name} validation confusion matrix at 0.50")
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)
    paths.append(confusion_path)
    return paths


def write_model_comparison_report(
    results: list[SearchResult],
    validation_prevalence: float,
    grids: dict[str, dict[str, list[Any]]],
    report_path: Path,
    tracking_uri: str,
) -> None:
    """Write the fixed-validation model comparison report."""
    ranked = sorted(results, key=lambda result: result.metrics.pr_auc, reverse=True)
    winner = ranked[0]
    rows = [
        "| Model | Best parameters | PR-AUC | ROC-AUC | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in ranked:
        metrics = result.metrics
        rows.append(
            f"| {result.model_name} | `{json.dumps(result.best_parameters, sort_keys=True)}` | "
            f"{metrics.pr_auc:.6f} | {metrics.roc_auc:.6f} | {metrics.precision:.6f} | "
            f"{metrics.recall:.6f} | {metrics.f1:.6f} |"
        )

    grid_lines = "\n".join(
        f"- **{model}:** `{json.dumps(grid, sort_keys=True)}`" for model, grid in grids.items()
    )
    run_lines = "\n".join(
        f"- **{result.model_name}:** best run `{result.best_run_id}`" for result in ranked
    )
    report = f"""# Baseline Model Comparison

## Experimental design

Three baselines were fitted on the chronological training split and selected on the immediately
following validation split: Logistic Regression, LightGBM, and XGBoost. The holdout set was not
used for fitting, preprocessing, parameter selection, feature selection, or evaluation.

Numeric medians and categorical frequencies were learned from training data only. Categories not
seen during training map to zero frequency. `isFraud` and `TransactionID` were excluded;
`TransactionDT` remains as relative time. No target encoding, SMOTE, or validation resampling was
used.

Validation fraud prevalence and no-skill PR-AUC: **{validation_prevalence:.6f}**

## Parameter grids

{grid_lines}

The boosting grids are deliberately limited to four configurations each because the dataset has
413k training rows and 432 baseline predictors. Logistic Regression uses one full-data baseline
configuration because repeated SAGA fits were disproportionately expensive; no rows were sampled.

## Best validation result per model

{chr(10).join(rows)}

## Winner

**BEST VALIDATION MODEL: {winner.model_name}**

It won because it achieved the highest validation PR-AUC ({winner.metrics.pr_auc:.6f}), the primary
metric for this imbalanced fraud-ranking problem. Threshold metrics use 0.50 only and have not been
optimized for business cost.

## MLflow

- Experiment: `fraud-detection-model-comparison`
- Tracking URI: `{tracking_uri}`
{run_lines}

Every grid configuration logs parameters, row counts, feature count, ranking metrics, and threshold
metrics. The best fitted estimator for each model is logged to its winning run.

## Known limitations

- This is baseline preprocessing, not final feature engineering.
- Frequency encoding can lose category-specific information, especially for the linear model.
- The logged estimator artifacts expect already-preprocessed matrices; preprocessing is not yet
  packaged into a production inference pipeline.
- Probabilities are not calibrated and the 0.50 threshold is not cost-optimized.
- No holdout metrics are reported. The holdout remains reserved for final evaluation and later
  production simulation.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
