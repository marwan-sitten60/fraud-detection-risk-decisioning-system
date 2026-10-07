import numpy as np
import pandas as pd
import pytest

from fraud_detection.models import (
    evaluate_catboost_fold,
    make_expanding_time_folds,
    prepare_catboost_data,
)


def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, 13),
            "TransactionDT": np.arange(100, 1300, 100),
            "TransactionAmt": np.arange(10.0, 130.0, 10.0),
            "card1": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "addr1": [10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60],
            "DeviceInfo": ["mobile", None] * 6,
            "P_emaildomain": ["a.test", "b.test"] * 6,
            "numeric": np.linspace(0.0, 1.0, 12),
            "isFraud": [0, 1] * 6,
        }
    )


def test_preparation_excludes_target_and_transaction_id() -> None:
    frame = transactions()
    prepared = prepare_catboost_data(frame.iloc[:8], frame.iloc[8:])

    assert "isFraud" not in prepared.train.columns
    assert "TransactionID" not in prepared.train.columns


def test_categorical_columns_and_missing_values_are_handled() -> None:
    frame = transactions()
    prepared = prepare_catboost_data(frame.iloc[:8], frame.iloc[8:])

    assert {"card1", "addr1", "DeviceInfo", "P_emaildomain"} <= set(prepared.categorical_names)
    assert "numeric" not in prepared.categorical_names
    assert "__MISSING__" in set(prepared.train["DeviceInfo"])


def test_unseen_evaluation_category_does_not_break_catboost() -> None:
    frame = transactions()
    train = frame.iloc[:8]
    evaluation = frame.iloc[8:].copy()
    evaluation["DeviceInfo"] = "unseen-device"
    prepared = prepare_catboost_data(train, evaluation)
    parameters = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC:type=Classic",
        "iterations": 10,
        "depth": 3,
        "learning_rate": 0.1,
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "allow_writing_files": False,
        "thread_count": 1,
    }

    result = evaluate_catboost_fold(
        parameters,
        prepared,
        train["isFraud"],
        evaluation["isFraud"],
        early_stopping_rounds=3,
    )

    assert 0 <= result.pr_auc <= 1


def test_historical_amount_features_remain_past_only() -> None:
    frame = transactions()
    prepared = prepare_catboost_data(frame.iloc[:8], frame.iloc[8:])

    assert (
        prepared.train.loc[0, "card1_previous_avg_amount"]
        != prepared.train.loc[0, "card1_previous_avg_amount"]
    )
    assert prepared.train.loc[1, "card1_previous_avg_amount"] == pytest.approx(10.0)
    assert (
        prepared.evaluation.iloc[0]["card1_previous_avg_amount"]
        != prepared.evaluation.iloc[0]["card1_previous_avg_amount"]
    )


def test_temporal_folds_remain_strictly_chronological() -> None:
    frame = transactions()
    folds = make_expanding_time_folds(frame, n_splits=3, initial_train_fraction=0.5)

    for fold in folds:
        train_times = frame.iloc[fold.train_indices]["TransactionDT"]
        evaluation_times = frame.iloc[fold.evaluation_indices]["TransactionDT"]
        assert train_times.max() < evaluation_times.min()
