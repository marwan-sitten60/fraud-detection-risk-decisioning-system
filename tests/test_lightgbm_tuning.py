import numpy as np
import pandas as pd
import pytest

from fraud_detection.models import (
    evaluate_lightgbm_fold,
    make_expanding_time_folds,
    summarize_lightgbm_cv,
)


def chronological_frame(row_count: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": np.arange(row_count),
            "TransactionDT": np.repeat(np.arange(row_count // 2), 2),
        }
    )


def test_expanding_folds_are_chronological_and_non_overlapping() -> None:
    frame = chronological_frame()
    folds = make_expanding_time_folds(frame, n_splits=3, initial_train_fraction=0.70)

    assert len(folds) == 3
    for fold in folds:
        assert np.intersect1d(fold.train_indices, fold.evaluation_indices).size == 0
        train_times = frame.iloc[fold.train_indices]["TransactionDT"]
        evaluation_times = frame.iloc[fold.evaluation_indices]["TransactionDT"]
        assert train_times.max() < evaluation_times.min()


def test_expanding_train_period_grows_each_fold() -> None:
    folds = make_expanding_time_folds(chronological_frame())

    assert [len(fold.train_indices) for fold in folds] == sorted(
        len(fold.train_indices) for fold in folds
    )
    assert len({len(fold.train_indices) for fold in folds}) == len(folds)


@pytest.mark.parametrize(
    ("frame", "kwargs", "message"),
    [
        (pd.DataFrame(), {}, "must not be empty"),
        (pd.DataFrame({"x": [1, 2]}), {}, "requires TransactionDT"),
        (chronological_frame(), {"n_splits": 1}, "at least 2"),
        (chronological_frame(), {"initial_train_fraction": 1.0}, "between 0 and 1"),
        (
            chronological_frame().sort_values("TransactionDT", ascending=False),
            {},
            "sorted chronologically",
        ),
    ],
)
def test_invalid_temporal_cv_configuration_raises(
    frame: pd.DataFrame, kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_expanding_time_folds(frame, **kwargs)


def test_lightgbm_fold_evaluation_is_deterministic() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 3)).astype("float32")
    y = (x[:, 0] + 0.25 * x[:, 1] > 0).astype(int)
    parameters = {
        "learning_rate": 0.1,
        "n_estimators": 40,
        "num_leaves": 7,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }

    first = evaluate_lightgbm_fold(
        parameters, x[:80], y[:80], x[80:], y[80:], early_stopping_rounds=5
    )
    second = evaluate_lightgbm_fold(
        parameters, x[:80], y[:80], x[80:], y[80:], early_stopping_rounds=5
    )

    assert first == second
    summary = summarize_lightgbm_cv(parameters, [first, second])
    assert summary.mean_pr_auc == pytest.approx(first.pr_auc)
    assert summary.std_pr_auc == pytest.approx(0.0)
