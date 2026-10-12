import numpy as np
import pandas as pd

from fraud_detection.inference.features import FrozenFeatureState
from fraud_detection.models import prepare_engineered_matrices


def test_persistable_feature_path_matches_research_builder_exactly() -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 11),
            "TransactionDT": [1, 1, 2, 3, 4, 5, 6, 7, 7, 9],
            "TransactionAmt": [10.0, 12.0, 8.0, 20.0, 40.0, 5.0, 30.0, 15.0, 18.0, 25.0],
            "card1": [1, 1, 1, 2, 1, 3, 2, 1, 4, 1],
            "card2": [1, 1, 1, 2, 1, 3, 2, 1, 4, 1],
            "addr1": [10, 10, 10, 20, 10, 30, 20, 10, 40, 10],
            "P_emaildomain": ["a", "a", "a", "b", "a", None, "b", "a", "c", "a"],
            "R_emaildomain": ["x", "x", "x", "y", "x", None, "y", "x", "z", "x"],
            "DeviceInfo": ["d", "d", "d", "e", "d", None, "e", "d", "f", "d"],
            "ProductCD": ["W", "W", "W", "C", "W", "R", "C", "W", "H", "W"],
            "isFraud": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        }
    )
    train = frame.iloc[:7]
    future = frame.iloc[7:].drop(columns="isFraud")
    reference = prepare_engineered_matrices(
        train,
        future,
        include_graph=False,
        include_composite=True,
        include_target_encoding=True,
        scaling_batch_rows=len(train),
    )
    state, train_matrix = FrozenFeatureState.fit_train(train)
    future_matrix = state.transform(future)
    positions = {name: index for index, name in enumerate(reference.feature_names)}
    indices = [positions[name] for name in state.feature_names]
    np.testing.assert_array_equal(train_matrix, reference.x_train[:, indices])
    np.testing.assert_array_equal(future_matrix, reference.x_evaluation[:, indices])
    assert "isFraud" not in state.raw_columns
    sequential_state, _ = FrozenFeatureState.fit_train(train)
    sequential = np.vstack(
        [sequential_state.transform(future.iloc[[position]]) for position in range(len(future))]
    )
    np.testing.assert_array_equal(future_matrix, sequential)
