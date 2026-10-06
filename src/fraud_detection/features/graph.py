"""Memory-conscious, past-only relationship features for transaction entities."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

CARD_DEVICE_FEATURES = (
    "card_previous_unique_devices",
    "device_previous_unique_cards",
    "card_device_previous_pair_count",
    "card_device_seen_before",
)
CARD_ADDRESS_FEATURES = (
    "card_previous_unique_addresses",
    "address_previous_unique_cards",
    "card_address_previous_pair_count",
    "card_address_seen_before",
)


@dataclass(slots=True)
class _RelationshipState:
    """Sparse adjacency and edge-count state for one bipartite relationship."""

    left_neighbors: dict[Any, set[Any]] = field(default_factory=dict)
    right_neighbors: dict[Any, set[Any]] = field(default_factory=dict)
    pair_counts: dict[tuple[Any, Any], int] = field(default_factory=dict)

    def features(self, left: Any | None, right: Any | None) -> tuple[int, int, int, int]:
        left_degree = len(self.left_neighbors.get(left, ())) if left is not None else 0
        right_degree = len(self.right_neighbors.get(right, ())) if right is not None else 0
        if left is None or right is None:
            return left_degree, right_degree, 0, 0
        pair_count = self.pair_counts.get((left, right), 0)
        return left_degree, right_degree, pair_count, int(pair_count > 0)

    def update(self, left: Any | None, right: Any | None) -> None:
        if left is None or right is None:
            return
        self.left_neighbors.setdefault(left, set()).add(right)
        self.right_neighbors.setdefault(right, set()).add(left)
        pair = (left, right)
        self.pair_counts[pair] = self.pair_counts.get(pair, 0) + 1


class GraphFeatureEngineer:
    """Generate target-independent graph features from strictly earlier timestamps.

    Rows at one ``TransactionDT`` are scored from the state left by earlier
    timestamps. The entire timestamp batch is committed only after all its rows
    have been scored, preventing both current-row and same-time leakage.

    Missing entity values are unavailable endpoints: they do not create nodes or
    edges. Known endpoints can still expose their prior degree, while pair-specific
    values are zero when either endpoint is missing.
    """

    def __init__(
        self,
        *,
        include_card_device: bool = True,
        include_card_address: bool = True,
    ) -> None:
        if not include_card_device and not include_card_address:
            raise ValueError("At least one graph relationship must be enabled")
        self.include_card_device = include_card_device
        self.include_card_address = include_card_address
        self.card_device_state_ = _RelationshipState()
        self.card_address_state_ = _RelationshipState()
        self.feature_groups_: dict[str, tuple[str, ...]] = {}
        self.generated_feature_names_: tuple[str, ...] = ()
        self.latest_timestamp_: float | None = None
        self.is_fitted_ = False

    def fit(self, dataframe: pd.DataFrame) -> "GraphFeatureEngineer":
        """Build graph state from a historical period without retaining features."""
        self._reset_state()
        self._validate_input(dataframe)
        self._process(dataframe)
        self.latest_timestamp_ = float(dataframe["TransactionDT"].max())
        self.is_fitted_ = True
        LOGGER.info("Initialized graph state from %d transactions", len(dataframe))
        return self

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with leakage-safe training graph features appended."""
        features = self.fit_transform_features(dataframe)
        return self._append_features(dataframe, features)

    def fit_transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return only training graph features and preserve final training state."""
        self._reset_state()
        self._validate_input(dataframe)
        features = self._process(dataframe)
        self.latest_timestamp_ = float(dataframe["TransactionDT"].max())
        self.is_fitted_ = True
        LOGGER.info(
            "Generated %d graph features for %d training transactions",
            len(self.generated_feature_names_),
            len(dataframe),
        )
        return features

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Score and then commit a later chronological batch to fitted graph state."""
        features = self.transform_features(dataframe)
        return self._append_features(dataframe, features)

    def transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return sequential graph features and update state after each timestamp batch."""
        if not self.is_fitted_:
            raise RuntimeError("GraphFeatureEngineer must be fitted before transform")
        self._validate_input(dataframe)
        current_minimum = float(dataframe["TransactionDT"].min())
        if self.latest_timestamp_ is not None and current_minimum <= self.latest_timestamp_:
            raise ValueError(
                "New graph-feature batches must start strictly after fitted graph history"
            )
        features = self._process(dataframe)
        self.latest_timestamp_ = float(dataframe["TransactionDT"].max())
        LOGGER.info("Generated sequential graph features for %d transactions", len(dataframe))
        return features

    @property
    def relationship_counts(self) -> dict[str, int]:
        """Return compact state cardinalities for monitoring and memory reporting."""
        return {
            "card_device_edges": len(self.card_device_state_.pair_counts),
            "card_address_edges": len(self.card_address_state_.pair_counts),
            "cards_with_devices": len(self.card_device_state_.left_neighbors),
            "devices_with_cards": len(self.card_device_state_.right_neighbors),
            "cards_with_addresses": len(self.card_address_state_.left_neighbors),
            "addresses_with_cards": len(self.card_address_state_.right_neighbors),
        }

    def _reset_state(self) -> None:
        self.card_device_state_ = _RelationshipState()
        self.card_address_state_ = _RelationshipState()
        self.latest_timestamp_ = None
        groups: dict[str, tuple[str, ...]] = {}
        if self.include_card_device:
            groups["card_device"] = CARD_DEVICE_FEATURES
        if self.include_card_address:
            groups["card_address"] = CARD_ADDRESS_FEATURES
        self.feature_groups_ = groups
        self.generated_feature_names_ = tuple(
            feature for features in groups.values() for feature in features
        )

    def _process(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        row_count = len(dataframe)
        output = {
            feature: np.zeros(row_count, dtype=np.int32)
            for feature in self.generated_feature_names_
        }
        order_frame = pd.DataFrame(
            {
                "TransactionDT": pd.to_numeric(dataframe["TransactionDT"], errors="raise"),
                "TransactionID": dataframe["TransactionID"].to_numpy(),
                "__position": np.arange(row_count),
            }
        ).sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
        sorted_positions = order_frame["__position"].to_numpy()
        sorted_times = order_frame["TransactionDT"].to_numpy()
        cards = dataframe["card1"].to_numpy()
        devices = dataframe["DeviceInfo"].to_numpy() if self.include_card_device else None
        addresses = dataframe["addr1"].to_numpy() if self.include_card_address else None

        boundaries = np.flatnonzero(np.r_[True, sorted_times[1:] != sorted_times[:-1], True])
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            positions = sorted_positions[start:end]
            for position in positions:
                card = self._available(cards[position])
                if self.include_card_device and devices is not None:
                    device = self._available(devices[position])
                    values = self.card_device_state_.features(card, device)
                    for feature, value in zip(CARD_DEVICE_FEATURES, values, strict=True):
                        output[feature][position] = value
                if self.include_card_address and addresses is not None:
                    address = self._available(addresses[position])
                    values = self.card_address_state_.features(card, address)
                    for feature, value in zip(CARD_ADDRESS_FEATURES, values, strict=True):
                        output[feature][position] = value

            for position in positions:
                card = self._available(cards[position])
                if self.include_card_device and devices is not None:
                    self.card_device_state_.update(card, self._available(devices[position]))
                if self.include_card_address and addresses is not None:
                    self.card_address_state_.update(card, self._available(addresses[position]))

        return pd.DataFrame(output, index=dataframe.index)

    def _validate_input(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            raise ValueError("Graph feature input must not be empty")
        required = ["TransactionID", "TransactionDT", "card1"]
        if self.include_card_device:
            required.append("DeviceInfo")
        if self.include_card_address:
            required.append("addr1")
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Graph feature input is missing required columns: {missing}")
        if dataframe["TransactionID"].duplicated().any():
            raise ValueError("TransactionID must be unique")
        times = pd.to_numeric(dataframe["TransactionDT"], errors="raise")
        if times.isna().any():
            raise ValueError("TransactionDT must not contain missing values")

    @staticmethod
    def _available(value: Any) -> Any | None:
        return None if pd.isna(value) else value

    @staticmethod
    def _append_features(dataframe: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        transformed = dataframe.copy()
        for feature in features.columns:
            transformed[feature] = features[feature].to_numpy()
        return transformed
