"""Probability calibration for frozen ensemble scores."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPSILON = 1e-6
REQUIRED_OOF = ("TransactionID", "fold", "y_true", "lightgbm_score", "xgboost_score")


def probabilities(values):
    """Validate a one-dimensional probability vector without modifying input."""
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Probabilities must be a nonempty finite one-dimensional vector")
    if ((result < 0) | (result > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    return result.copy()


def targets(values):
    result = np.asarray(values)
    if result.ndim != 1 or not len(result) or not np.isin(result, (0, 1)).all():
        raise ValueError("Targets must be a nonempty binary vector")
    return result.astype(int, copy=True)


def validate_oof(frame):
    """Require exact, ordered temporal folds and unique aligned row identifiers."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("OOF input must be a DataFrame")
    missing = set(REQUIRED_OOF) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OOF columns: {sorted(missing)}")
    if frame.empty or frame.loc[:, list(REQUIRED_OOF)].isna().any().any():
        raise ValueError("OOF columns must be nonempty and complete")
    if frame.TransactionID.duplicated().any():
        raise ValueError("OOF row identifiers must be unique")
    if frame.fold.tolist() != sorted(frame.fold.tolist()) or set(frame.fold) != {1, 2, 3}:
        raise ValueError("OOF rows must be ordered in chronological folds 1, 2, 3")
    if (
        not frame.groupby("fold", sort=True)
        .TransactionID.apply(lambda x: x.is_monotonic_increasing)
        .all()
    ):
        raise ValueError("OOF row identifiers must increase within each fold")
    y = targets(frame.y_true)
    lightgbm = probabilities(frame.lightgbm_score)
    xgboost = probabilities(frame.xgboost_score)
    if not (len(y) == len(lightgbm) == len(xgboost)):
        raise ValueError("OOF arrays are not aligned")
    return 0.8 * lightgbm + 0.2 * xgboost


@dataclass
class Calibrator:
    """Serializable Platt-on-logit or isotonic probability mapping."""

    method: str
    model: object = None

    def fit(self, score, label):
        p, y = probabilities(score), targets(label)
        if len(p) != len(y) or len(np.unique(y)) != 2:
            raise ValueError("Calibration fit requires aligned scores and both target classes")
        if self.method == "platt":
            x = np.log(
                np.clip(p, EPSILON, 1 - EPSILON) / (1 - np.clip(p, EPSILON, 1 - EPSILON))
            ).reshape(-1, 1)
            self.model = LogisticRegression(penalty=None, max_iter=1000).fit(x, y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        else:
            raise ValueError("Method must be platt or isotonic")
        return self

    def transform(self, score):
        p = probabilities(score)
        if self.model is None:
            raise ValueError("Calibrator has not been fitted")
        if self.method == "platt":
            x = np.log(
                np.clip(p, EPSILON, 1 - EPSILON) / (1 - np.clip(p, EPSILON, 1 - EPSILON))
            ).reshape(-1, 1)
            return np.clip(self.model.predict_proba(x)[:, 1], 0, 1)
        return np.clip(self.model.predict(p), 0, 1)


def reliability_bins(label, score, n_bins=10):
    """Equal-frequency bins; ECE weights absolute bin gaps by row share."""
    y, p = targets(label), probabilities(score)
    if len(y) != len(p) or n_bins < 1:
        raise ValueError("Invalid reliability inputs")
    order = np.argsort(p, kind="mergesort")
    rows = []
    for number, positions in enumerate(np.array_split(order, min(n_bins, len(p))), 1):
        rows.append(
            {
                "bin": number,
                "count": len(positions),
                "score_min": float(p[positions].min()),
                "score_max": float(p[positions].max()),
                "mean_probability": float(p[positions].mean()),
                "fraud_rate": float(y[positions].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(label, score, n_bins=10):
    y, p = targets(label), probabilities(score)
    if len(y) != len(p):
        raise ValueError("Scores and targets must align")
    bins = reliability_bins(y, p, n_bins)
    result = {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(
            np.average(abs(bins.mean_probability - bins.fraud_rate), weights=bins["count"])
        ),
        "mean_probability": float(p.mean()),
        "prevalence": float(y.mean()),
        "calibration_gap": float(p.mean() - y.mean()),
    }
    if len(np.unique(y)) == 2:
        result.update(
            pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p))
        )
    return result


def temporal_selection(oof):
    """Evaluate raw, Platt, isotonic on fold 2 after 1 and fold 3 after 1+2."""
    score = validate_oof(oof)
    y = targets(oof.y_true)
    fold = oof.fold.to_numpy()
    rows = []
    predictions = {}
    for method in ("raw", "platt", "isotonic"):
        pieces = []
        for evaluation_fold in (2, 3):
            fit = fold < evaluation_fold
            test = fold == evaluation_fold
            predicted = (
                score[test]
                if method == "raw"
                else Calibrator(method).fit(score[fit], y[fit]).transform(score[test])
            )
            metrics = evaluate(y[test], predicted)
            rows.append(
                {
                    "method": method,
                    "evaluation_fold": evaluation_fold,
                    "fit_rows": int(fit.sum()),
                    "evaluation_rows": int(test.sum()),
                    **metrics,
                }
            )
            pieces.append((y[test], predicted))
        pooled_y = np.concatenate([part[0] for part in pieces])
        pooled_p = np.concatenate([part[1] for part in pieces])
        predictions[method] = (pooled_y, pooled_p)
        rows.append(
            {
                "method": method,
                "evaluation_fold": "pooled",
                "fit_rows": np.nan,
                "evaluation_rows": len(pooled_y),
                **evaluate(pooled_y, pooled_p),
            }
        )
    return pd.DataFrame(rows), predictions


def select_method(results, tolerance=1e-5):
    pooled = results.loc[results.evaluation_fold == "pooled"].set_index("method")
    best = min(("raw", "platt", "isotonic"), key=lambda method: pooled.loc[method, "brier"])
    if (
        best == "isotonic"
        and pooled.loc["platt", "brier"] - pooled.loc["isotonic", "brier"] <= tolerance
    ):
        return "platt"
    return best
