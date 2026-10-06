import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import BehavioralFeatureEngineer


def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100, 200, 300, 400],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "card1": ["A", "A", "A", "B"],
            "addr1": [1, 1, 2, 1],
            "DeviceInfo": ["mobile", "mobile", "desktop", None],
            "P_emaildomain": ["mail.test", "mail.test", None, "other.test"],
            "isFraud": [0, 1, 0, 1],
        }
    )


def test_previous_counts_exclude_current_transaction() -> None:
    transformed = BehavioralFeatureEngineer().fit_transform(transactions())

    assert transformed["card1_previous_tx_count"].tolist() == [0, 1, 2, 0]


def test_historical_amounts_use_prior_rows_only() -> None:
    transformed = BehavioralFeatureEngineer().fit_transform(transactions())

    assert np.isnan(transformed.loc[0, "card1_previous_avg_amount"])
    assert transformed.loc[1, "card1_previous_avg_amount"] == pytest.approx(10.0)
    assert transformed.loc[2, "card1_previous_avg_amount"] == pytest.approx(15.0)
    assert transformed.loc[2, "card1_previous_max_amount"] == pytest.approx(20.0)


def test_velocity_excludes_current_and_equal_timestamp_transactions() -> None:
    frame = transactions().iloc[:3].copy()
    frame["TransactionDT"] = [100, 100, 200]
    transformed = BehavioralFeatureEngineer().fit_transform(frame)

    assert transformed["card1_tx_count_last_1h"].tolist() == [0, 0, 2]


def test_equal_timestamps_are_ordered_by_transaction_id() -> None:
    frame = transactions().iloc[:3].copy().iloc[[1, 0, 2]]
    frame["TransactionDT"] = [100, 100, 200]
    transformed = BehavioralFeatureEngineer().fit_transform(frame)

    by_id = transformed.set_index("TransactionID")
    assert by_id.loc[1, "card1_previous_tx_count"] == 0
    assert by_id.loc[2, "card1_previous_tx_count"] == 1
    assert by_id.loc[2, "seconds_since_card1_previous_transaction"] == 0


def test_validation_uses_training_and_earlier_validation_history() -> None:
    train = transactions().iloc[:2]
    validation = transactions().iloc[2:].copy()
    validation.loc[:, "card1"] = ["A", "A"]
    engineer = BehavioralFeatureEngineer().fit(train)

    transformed = engineer.transform(validation)

    assert transformed["card1_previous_tx_count"].tolist() == [2, 3]
    assert transformed["card1_previous_avg_amount"].tolist() == pytest.approx([15.0, 20.0])


def test_unseen_validation_entity_starts_with_no_history() -> None:
    train = transactions().iloc[:2]
    validation = transactions().iloc[[3]].copy()
    validation.loc[:, "card1"] = "NEW"
    transformed = BehavioralFeatureEngineer().fit(train).transform(validation)

    assert transformed.loc[validation.index[0], "card1_previous_tx_count"] == 0
    assert np.isnan(transformed.loc[validation.index[0], "card1_previous_avg_amount"])


def test_missing_entities_share_a_safe_explicit_history_bucket() -> None:
    frame = transactions().iloc[:3].copy()
    frame["card1"] = [None, None, "A"]
    transformed = BehavioralFeatureEngineer().fit_transform(frame)

    assert transformed["card1_previous_tx_count"].tolist() == [0, 1, 0]


def test_future_transaction_changes_do_not_affect_earlier_features() -> None:
    original = transactions()
    changed = original.copy()
    changed.loc[2, ["TransactionAmt", "card1"]] = [9999.0, "FUTURE"]

    original_features = BehavioralFeatureEngineer().fit_transform(original)
    changed_features = BehavioralFeatureEngineer().fit_transform(changed)

    feature_names = BehavioralFeatureEngineer().fit(original).generated_feature_names_
    pd.testing.assert_series_equal(
        original_features.loc[1, list(feature_names)],
        changed_features.loc[1, list(feature_names)],
    )


def test_target_never_affects_behavioral_features() -> None:
    original = transactions()
    changed = original.copy()
    changed["isFraud"] = 1 - changed["isFraud"]
    original_features = BehavioralFeatureEngineer().fit_transform(original)
    changed_features = BehavioralFeatureEngineer().fit_transform(changed)
    feature_names = BehavioralFeatureEngineer().fit(original).generated_feature_names_

    pd.testing.assert_frame_equal(
        original_features.loc[:, list(feature_names)],
        changed_features.loc[:, list(feature_names)],
    )


def test_input_is_not_mutated() -> None:
    frame = transactions()
    original = frame.copy(deep=True)
    engineer = BehavioralFeatureEngineer()

    engineer.fit_transform(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_optional_entity_columns_may_be_absent() -> None:
    frame = transactions().drop(columns=["addr1", "DeviceInfo", "P_emaildomain"])
    transformed = BehavioralFeatureEngineer().fit_transform(frame)

    assert "card1_previous_tx_count" in transformed
    assert "addr1_previous_tx_count" not in transformed
