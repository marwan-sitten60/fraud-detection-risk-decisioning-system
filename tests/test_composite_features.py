import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import CompositeFeatureEngineer, create_composite_keys


def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [100, 100, 200, 300, 400],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
            "card1": ["A", "A", "A", "A", "B"],
            "card2": [1, 1, 1, 1, 2],
            "addr1": [10, 10, 10, 10, 20],
            "P_emaildomain": ["mail.test", "mail.test", "mail.test", "mail.test", None],
            "ProductCD": ["W", "W", "W", "W", "C"],
            "DeviceInfo": ["mobile", "mobile", "mobile", "mobile", None],
            "isFraud": [0, 1, 0, 1, 0],
        }
    )


def test_composite_keys_are_deterministic_and_unambiguous() -> None:
    frame = transactions()

    first = create_composite_keys(frame)
    second = create_composite_keys(frame.copy())

    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0, 0] == "V1:A|V1:1|V2:10"


def test_missing_components_use_explicit_bucket_without_string_collision() -> None:
    frame = transactions().iloc[[0, 4]].copy()
    frame.loc[frame.index[0], "P_emaildomain"] = "M"

    keys = create_composite_keys(frame)

    assert keys.iloc[0]["card1_addr1_p_emaildomain"].endswith("V1:M")
    assert keys.iloc[1]["card1_addr1_p_emaildomain"].endswith("M")
    assert keys.iloc[0]["card1_addr1_p_emaildomain"] != keys.iloc[1]["card1_addr1_p_emaildomain"]


def test_history_is_strictly_past_only_and_same_timestamp_is_isolated() -> None:
    transformed = CompositeFeatureEngineer().fit_transform_features(transactions())
    prefix = "card1_card2_addr1"

    assert transformed[f"{prefix}_previous_tx_count"].tolist() == [0, 0, 2, 3, 0]
    assert np.isnan(transformed.loc[0, f"{prefix}_previous_avg_amount"])
    assert np.isnan(transformed.loc[1, f"{prefix}_previous_avg_amount"])
    assert transformed.loc[2, f"{prefix}_previous_avg_amount"] == pytest.approx(15.0)
    assert transformed.loc[2, f"{prefix}_previous_max_amount"] == pytest.approx(20.0)
    assert transformed.loc[2, f"{prefix}_amount_vs_previous_avg"] == pytest.approx(2.0)


def test_validation_uses_train_and_only_earlier_timestamp_batches() -> None:
    frame = transactions()
    train = frame.iloc[:3]
    evaluation = frame.iloc[3:5].copy()
    evaluation.loc[:, "card1"] = "A"
    evaluation.loc[:, "card2"] = 1
    evaluation.loc[:, "addr1"] = 10
    evaluation.loc[:, "P_emaildomain"] = "mail.test"
    evaluation.loc[:, "ProductCD"] = "W"
    evaluation.loc[:, "DeviceInfo"] = "mobile"
    evaluation.loc[:, "TransactionDT"] = 300

    transformed = CompositeFeatureEngineer().fit(train).transform_features(evaluation)
    prefix = "card1_card2_addr1"

    assert transformed[f"{prefix}_previous_tx_count"].tolist() == [3, 3]
    assert transformed[f"{prefix}_previous_avg_amount"].tolist() == pytest.approx([20.0, 20.0])


def test_future_changes_do_not_affect_earlier_features() -> None:
    original = transactions()
    changed = original.copy()
    changed.loc[4, ["TransactionAmt", "card1", "card2", "addr1"]] = [9999.0, "Z", 9, 99]

    original_features = CompositeFeatureEngineer().fit_transform_features(original)
    changed_features = CompositeFeatureEngineer().fit_transform_features(changed)
    history_columns = CompositeFeatureEngineer().feature_groups_["composite_history"]

    pd.testing.assert_frame_equal(
        original_features.loc[:3, list(history_columns)],
        changed_features.loc[:3, list(history_columns)],
    )


def test_unseen_evaluation_identity_has_zero_frequency_and_empty_history() -> None:
    frame = transactions()
    train = frame.iloc[:4]
    evaluation = frame.iloc[[4]]

    transformed = CompositeFeatureEngineer().fit(train).transform_features(evaluation)

    assert transformed.filter(like="_composite_frequency").eq(0.0).all().all()
    assert transformed.filter(like="_previous_tx_count").eq(0).all().all()


def test_frequency_maps_are_fitted_only_on_reference_rows() -> None:
    frame = transactions()
    train = frame.iloc[:3]
    evaluation = frame.iloc[3:].copy()
    evaluation.loc[:, "card1"] = ["A", "NEW"]

    transformed = CompositeFeatureEngineer().fit(train).transform_features(evaluation)
    feature = "card1_card2_addr1_composite_frequency"

    assert transformed.iloc[0][feature] == pytest.approx(1.0)
    assert transformed.iloc[1][feature] == pytest.approx(0.0)


def test_target_is_independent_and_input_is_not_mutated() -> None:
    frame = transactions()
    original = frame.copy(deep=True)
    changed = frame.copy()
    changed["isFraud"] = 1 - changed["isFraud"]

    original_features = CompositeFeatureEngineer().fit_transform_features(frame)
    changed_features = CompositeFeatureEngineer().fit_transform_features(changed)

    pd.testing.assert_frame_equal(original_features, changed_features)
    pd.testing.assert_frame_equal(frame, original)
