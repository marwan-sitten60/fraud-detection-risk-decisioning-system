import pandas as pd
import pytest

from fraud_detection.features import GraphFeatureEngineer


def graph_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100, 200, 300, 400],
            "card1": ["A", "B", "A", "A"],
            "DeviceInfo": ["X", "X", "Y", "X"],
            "addr1": [10, 20, 10, 10],
            "isFraud": [0, 1, 0, 1],
        }
    )


def test_card_device_degree_growth_uses_previous_relationships_only() -> None:
    transformed = GraphFeatureEngineer().fit_transform(graph_transactions())

    assert transformed["card_previous_unique_devices"].tolist() == [0, 0, 1, 2]
    assert transformed["device_previous_unique_cards"].tolist() == [0, 1, 0, 2]


def test_pair_counts_increment_after_each_transaction() -> None:
    transformed = GraphFeatureEngineer().fit_transform(graph_transactions())

    assert transformed["card_device_previous_pair_count"].tolist() == [0, 0, 0, 1]
    assert transformed["card_device_seen_before"].tolist() == [0, 0, 0, 1]
    assert transformed["card_address_previous_pair_count"].tolist() == [0, 0, 1, 2]


def test_same_timestamp_transactions_are_isolated() -> None:
    frame = graph_transactions().iloc[:3].copy()
    frame["TransactionDT"] = [100, 100, 200]
    frame["card1"] = ["A", "B", "A"]
    frame["DeviceInfo"] = ["X", "X", "X"]
    transformed = GraphFeatureEngineer().fit_transform(frame)

    assert transformed["device_previous_unique_cards"].tolist() == [0, 0, 2]
    assert transformed["card_device_previous_pair_count"].tolist() == [0, 0, 1]


def test_future_changes_do_not_affect_earlier_features() -> None:
    original = graph_transactions()
    changed = original.copy()
    changed.loc[3, ["card1", "DeviceInfo", "addr1"]] = ["FUTURE", "Z", 999]
    original_features = GraphFeatureEngineer().fit_transform_features(original)
    changed_features = GraphFeatureEngineer().fit_transform_features(changed)

    pd.testing.assert_frame_equal(original_features.iloc[:3], changed_features.iloc[:3])


def test_validation_starts_from_train_state_and_updates_sequentially() -> None:
    train = graph_transactions().iloc[:2]
    validation = graph_transactions().iloc[2:].copy()
    engineer = GraphFeatureEngineer().fit(train)
    transformed = engineer.transform(validation)

    assert transformed["card_previous_unique_devices"].tolist() == [1, 2]
    assert transformed["card_device_previous_pair_count"].tolist() == [0, 1]


def test_validation_same_timestamp_rows_do_not_share_updates() -> None:
    train = graph_transactions().iloc[:1]
    validation = graph_transactions().iloc[1:3].copy()
    validation["TransactionDT"] = 200
    validation["card1"] = "A"
    validation["DeviceInfo"] = "Y"
    engineer = GraphFeatureEngineer().fit(train)
    transformed = engineer.transform(validation)

    assert transformed["card_device_previous_pair_count"].tolist() == [0, 0]


def test_new_batch_must_start_after_fitted_timestamp() -> None:
    train = graph_transactions().iloc[:1]
    validation = graph_transactions().iloc[[1]].copy()
    validation["TransactionDT"] = train["TransactionDT"].max()
    engineer = GraphFeatureEngineer().fit(train)

    with pytest.raises(ValueError, match="strictly after"):
        engineer.transform_features(validation)


def test_missing_entities_do_not_create_shared_nodes_or_edges() -> None:
    frame = graph_transactions().iloc[:3].copy()
    frame["DeviceInfo"] = None
    frame["addr1"] = None
    engineer = GraphFeatureEngineer()
    transformed = engineer.fit_transform(frame)

    assert transformed[list(engineer.generated_feature_names_)].to_numpy().sum() == 0
    assert engineer.relationship_counts["card_device_edges"] == 0
    assert engineer.relationship_counts["card_address_edges"] == 0


def test_known_card_degree_survives_missing_current_counterpart() -> None:
    frame = graph_transactions().iloc[:2].copy()
    frame.loc[1, "card1"] = "A"
    frame.loc[1, "DeviceInfo"] = None
    transformed = GraphFeatureEngineer().fit_transform(frame)

    assert transformed.loc[1, "card_previous_unique_devices"] == 1
    assert transformed.loc[1, "card_device_previous_pair_count"] == 0


def test_input_is_not_mutated() -> None:
    frame = graph_transactions()
    original = frame.copy(deep=True)
    GraphFeatureEngineer().fit_transform(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_target_values_do_not_affect_graph_features() -> None:
    original = graph_transactions()
    changed = original.copy()
    changed["isFraud"] = 1 - changed["isFraud"]
    original_features = GraphFeatureEngineer().fit_transform_features(original)
    changed_features = GraphFeatureEngineer().fit_transform_features(changed)

    pd.testing.assert_frame_equal(original_features, changed_features)


def test_relationship_groups_can_be_generated_independently() -> None:
    frame = graph_transactions()
    device_features = GraphFeatureEngineer(
        include_card_device=True, include_card_address=False
    ).fit_transform_features(frame.drop(columns="addr1"))
    address_features = GraphFeatureEngineer(
        include_card_device=False, include_card_address=True
    ).fit_transform_features(frame.drop(columns="DeviceInfo"))

    assert device_features.columns.tolist() == [
        "card_previous_unique_devices",
        "device_previous_unique_cards",
        "card_device_previous_pair_count",
        "card_device_seen_before",
    ]
    assert address_features.columns.tolist() == [
        "card_previous_unique_addresses",
        "address_previous_unique_cards",
        "card_address_previous_pair_count",
        "card_address_seen_before",
    ]
