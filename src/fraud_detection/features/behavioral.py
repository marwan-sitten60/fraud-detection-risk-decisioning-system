"""Leakage-safe historical behavior features for chronological transaction data."""

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MISSING_ENTITY = "<MISSING>"
BASE_ENTITY_SPECS = (
    ("card1", "card1"),
    ("addr1", "addr1"),
    ("DeviceInfo", "device"),
    ("P_emaildomain", "email"),
)


class BehavioralFeatureEngineer:
    """Create target-independent features from strictly prior transactions.

    Rows are ordered by ``TransactionDT`` and then ``TransactionID``. Historical
    aggregates use rows earlier in that deterministic order. Velocity features
    are stricter: their prior timestamp must satisfy ``T - window < time < T``,
    so transactions sharing the current timestamp are excluded.

    ``fit_transform`` generates leakage-safe training features and stores only
    non-target history. ``transform`` then simulates sequential validation or
    inference: stored history and earlier rows in the new batch are available,
    while the current and future rows are not.
    """

    def __init__(
        self,
        *,
        include_counts: bool = True,
        include_amounts: bool = True,
        include_recency: bool = True,
        include_velocity: bool = True,
    ) -> None:
        self.include_counts = include_counts
        self.include_amounts = include_amounts
        self.include_recency = include_recency
        self.include_velocity = include_velocity
        self.feature_groups_: dict[str, tuple[str, ...]] = {}
        self.generated_feature_names_: tuple[str, ...] = ()
        self.history_: pd.DataFrame | None = None
        self.is_fitted_ = False

    def fit(self, dataframe: pd.DataFrame) -> "BehavioralFeatureEngineer":
        """Initialize non-target historical state from a chronological period."""
        self._validate_input(dataframe)
        self._configure_features(dataframe.columns)
        self.history_ = self._history_columns(dataframe).copy()
        self.is_fitted_ = True
        LOGGER.info("Initialized behavioral history with %d transactions", len(dataframe))
        return self

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generate past-only training features and initialize historical state."""
        features = self.fit_transform_features(dataframe)
        return self._append_features(dataframe, features)

    def fit_transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return only past-only training features and initialize historical state."""
        self._validate_input(dataframe)
        self._configure_features(dataframe.columns)
        features = self._compute_features(self._history_columns(dataframe))
        features.index = dataframe.index
        self.history_ = self._history_columns(dataframe).copy()
        self.is_fitted_ = True
        LOGGER.info(
            "Generated %d behavioral features for %d training transactions",
            len(self.generated_feature_names_),
            len(dataframe),
        )
        return features

    def transform(self, dataframe: pd.DataFrame, *, update_state: bool = True) -> pd.DataFrame:
        """Generate sequential features from fitted history and optionally retain the batch."""
        features = self.transform_features(dataframe, update_state=update_state)
        return self._append_features(dataframe, features)

    def transform_features(
        self, dataframe: pd.DataFrame, *, update_state: bool = True
    ) -> pd.DataFrame:
        """Return only sequential behavioral features for a new chronological batch."""
        if not self.is_fitted_ or self.history_ is None:
            raise RuntimeError("BehavioralFeatureEngineer must be fitted before transform")
        self._validate_input(dataframe)
        current = self._history_columns(dataframe)
        self._validate_period_follows_history(current)
        combined = pd.concat([self.history_, current], ignore_index=True, copy=False)
        combined_features = self._compute_features(combined)
        current_features = combined_features.iloc[-len(dataframe) :].reset_index(drop=True)
        current_features.index = dataframe.index
        if update_state:
            self.history_ = combined
        LOGGER.info(
            "Generated behavioral features for %d sequential transactions; update_state=%s",
            len(dataframe),
            update_state,
        )
        return current_features

    def _configure_features(self, columns: Iterable[str]) -> None:
        available = set(columns)
        groups: dict[str, list[str]] = {
            "historical_counts": [],
            "historical_amounts": [],
            "recency": [],
            "velocity": [],
        }
        if self.include_counts:
            groups["historical_counts"].extend(
                f"{prefix}_previous_tx_count"
                for source, prefix in BASE_ENTITY_SPECS
                if source in available
            )
            if {"card1", "addr1"} <= available:
                groups["historical_counts"].append("card1_addr1_previous_tx_count")
        if self.include_amounts:
            if "card1" in available:
                groups["historical_amounts"].extend(
                    (
                        "card1_previous_avg_amount",
                        "card1_previous_max_amount",
                        "amount_vs_card1_previous_avg",
                    )
                )
            if "addr1" in available:
                groups["historical_amounts"].extend(
                    ("addr1_previous_avg_amount", "amount_vs_addr1_previous_avg")
                )
        if self.include_recency:
            if "card1" in available:
                groups["recency"].append("seconds_since_card1_previous_transaction")
            if "DeviceInfo" in available:
                groups["recency"].append("seconds_since_device_previous_transaction")
        if self.include_velocity and "card1" in available:
            groups["velocity"].extend(("card1_tx_count_last_1h", "card1_tx_count_last_24h"))

        self.feature_groups_ = {name: tuple(features) for name, features in groups.items()}
        self.generated_feature_names_ = tuple(
            feature for features in self.feature_groups_.values() for feature in features
        )

    def _compute_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        working = dataframe.reset_index(drop=True).copy()
        working["__position"] = np.arange(len(working))
        working = working.sort_values(
            ["TransactionDT", "TransactionID"], kind="mergesort", ignore_index=True
        )
        times = pd.to_numeric(working["TransactionDT"], errors="raise").astype("float64")
        amounts = pd.to_numeric(working["TransactionAmt"], errors="raise").astype("float64")
        entity_keys = self._entity_keys(working)
        features = pd.DataFrame(index=working.index)

        if self.include_counts:
            for prefix, keys in entity_keys.items():
                feature_name = f"{prefix}_previous_tx_count"
                if feature_name in self.generated_feature_names_:
                    features[feature_name] = (
                        keys.groupby(keys, sort=False).cumcount().astype("int32")
                    )

        if self.include_amounts:
            for prefix in ("card1", "addr1"):
                keys = entity_keys.get(prefix)
                average_name = f"{prefix}_previous_avg_amount"
                if keys is None or average_name not in self.generated_feature_names_:
                    continue
                previous_count = keys.groupby(keys, sort=False).cumcount().astype("float64")
                previous_sum = amounts.groupby(keys, sort=False).cumsum() - amounts
                previous_average = previous_sum.div(previous_count.where(previous_count > 0))
                features[average_name] = previous_average.astype("float32")
                if prefix == "card1":
                    inclusive_max = amounts.groupby(keys, sort=False).cummax()
                    features["card1_previous_max_amount"] = (
                        inclusive_max.groupby(keys, sort=False).shift(1).astype("float32")
                    )
                ratio_name = f"amount_vs_{prefix}_previous_avg"
                features[ratio_name] = self._safe_ratio(amounts, previous_average)

        if self.include_recency:
            for prefix in ("card1", "device"):
                keys = entity_keys.get(prefix)
                feature_name = f"seconds_since_{prefix}_previous_transaction"
                if keys is None or feature_name not in self.generated_feature_names_:
                    continue
                previous_time = times.groupby(keys, sort=False).shift(1)
                features[feature_name] = (times - previous_time).astype("float32")

        if "card1_tx_count_last_1h" in self.generated_feature_names_:
            features["card1_tx_count_last_1h"] = self._strict_window_counts(
                times, entity_keys["card1"], 3_600
            )
            features["card1_tx_count_last_24h"] = self._strict_window_counts(
                times, entity_keys["card1"], 86_400
            )

        features["__position"] = working["__position"]
        return features.sort_values("__position").drop(columns="__position").reset_index(drop=True)

    @staticmethod
    def _strict_window_counts(times: pd.Series, keys: pd.Series, window_seconds: int) -> np.ndarray:
        counts = np.zeros(len(times), dtype=np.int32)
        time_values = times.to_numpy()
        for positions in keys.groupby(keys, sort=False).indices.values():
            entity_times = time_values[positions]
            lower = np.searchsorted(entity_times, entity_times - window_seconds, side="right")
            upper = np.searchsorted(entity_times, entity_times, side="left")
            counts[positions] = upper - lower
        return counts

    @staticmethod
    def _safe_ratio(amounts: pd.Series, previous_average: pd.Series) -> pd.Series:
        result = np.full(len(amounts), np.nan, dtype=np.float32)
        denominator = previous_average.to_numpy()
        valid = np.isfinite(denominator) & (denominator != 0)
        result[valid] = (amounts.to_numpy()[valid] / denominator[valid]).astype(np.float32)
        return pd.Series(result, index=amounts.index)

    @staticmethod
    def _normalize_entity(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna(MISSING_ENTITY)

    def _entity_keys(self, dataframe: pd.DataFrame) -> dict[str, pd.Series]:
        keys = {
            prefix: self._normalize_entity(dataframe[source])
            for source, prefix in BASE_ENTITY_SPECS
            if source in dataframe.columns
        }
        if "card1" in keys and "addr1" in keys:
            keys["card1_addr1"] = keys["card1"] + "|" + keys["addr1"]
        return keys

    def _history_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        sources = ["TransactionID", "TransactionDT", "TransactionAmt"]
        sources.extend(source for source, _ in BASE_ENTITY_SPECS if source in dataframe.columns)
        return dataframe.loc[:, sources]

    @staticmethod
    def _append_features(dataframe: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        transformed = dataframe.copy()
        for feature in features.columns:
            transformed[feature] = features[feature].to_numpy()
        return transformed

    @staticmethod
    def _validate_input(dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            raise ValueError("Behavioral feature input must not be empty")
        required = ("TransactionID", "TransactionDT", "TransactionAmt")
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Behavioral feature input is missing required columns: {missing}")
        if dataframe["TransactionID"].duplicated().any():
            raise ValueError("TransactionID must be unique for deterministic behavioral ordering")
        times = pd.to_numeric(dataframe["TransactionDT"], errors="raise")
        amounts = pd.to_numeric(dataframe["TransactionAmt"], errors="raise")
        if times.isna().any():
            raise ValueError("TransactionDT must not contain missing values")
        if amounts.isna().any() or not np.isfinite(amounts).all():
            raise ValueError("TransactionAmt must contain finite, non-missing values")

    def _validate_period_follows_history(self, dataframe: pd.DataFrame) -> None:
        if self.history_ is None or self.history_.empty:
            return
        if dataframe["TransactionID"].isin(self.history_["TransactionID"]).any():
            raise ValueError("New transactions must not repeat a historical TransactionID")
        history_max_time = self.history_["TransactionDT"].max()
        current_min_time = dataframe["TransactionDT"].min()
        if current_min_time < history_max_time:
            raise ValueError("New transactions must not precede the fitted historical period")
