"""Leakage-safe temporal target encoding for fraud-risk entities."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.features.composite import (
    DEFAULT_COMPOSITE_IDENTITIES,
    create_composite_keys,
)

LOGGER = logging.getLogger(__name__)

BASE_TARGET_ENTITIES = (
    "card1",
    "card2",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
)
COMPOSITE_TARGET_ENTITIES = tuple(identity.name for identity in DEFAULT_COMPOSITE_IDENTITIES)
TARGET_ENCODING_FEATURES = tuple(
    f"{entity}_target_encoded_fraud_rate"
    for entity in (*BASE_TARGET_ENTITIES, *COMPOSITE_TARGET_ENTITIES)
)
MISSING_ENTITY = "<MISSING>"
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class LabelAvailabilityStatistics:
    """Descriptive statistics for legally matured labels at query timestamps."""

    row_count: int
    history_row_count: int
    fallback_row_count: int
    fallback_fraction: float
    mean_matured_label_count: float
    mean_matured_label_fraction: float
    representative_matured_fractions: tuple[float, float, float]
    average_entity_support: dict[str, float]


class TemporalTargetEncoder:
    """Create strict temporal-OOF encodings and frozen future-period maps.

    Training rows at timestamp ``T`` use labels only from timestamps strictly
    before ``T``. The earliest timestamp uses ``initial_prior`` because no legal
    historical label exists. ``fit`` stores plain dictionaries and scalars so
    the frozen state remains straightforward to serialize for production.
    """

    def __init__(
        self,
        *,
        alpha: float = 50.0,
        initial_prior: float = 0.5,
        label_delay_days: float = 0.0,
    ) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a positive finite number")
        if not np.isfinite(initial_prior) or not 0 <= initial_prior <= 1:
            raise ValueError("initial_prior must be between 0 and 1")
        if not np.isfinite(label_delay_days) or label_delay_days < 0:
            raise ValueError("label_delay_days must be a non-negative finite number")
        self.alpha = float(alpha)
        self.initial_prior = float(initial_prior)
        self.label_delay_days = float(label_delay_days)
        self.label_delay_seconds = self.label_delay_days * SECONDS_PER_DAY
        self.feature_names_ = TARGET_ENCODING_FEATURES
        self.global_fraud_rate_: float | None = None
        self.entity_counts_: dict[str, dict[str, int]] = {}
        self.entity_fraud_counts_: dict[str, dict[str, int]] = {}
        self.encoding_maps_: dict[str, dict[str, float]] = {}
        self.training_statistics_: LabelAvailabilityStatistics | None = None
        self.last_transform_statistics_: LabelAvailabilityStatistics | None = None
        self._fit_history_: pd.DataFrame | None = None
        self.is_fitted_ = False

    def fit(self, dataframe: pd.DataFrame) -> "TemporalTargetEncoder":
        """Fit frozen maps using all labels in one legally available fit period."""
        self._validate_input(dataframe, require_target=True)
        targets = self._targets(dataframe)
        keys = self._entity_keys(dataframe)
        self.global_fraud_rate_ = float(targets.mean())
        self.entity_counts_ = {}
        self.entity_fraud_counts_ = {}
        self.encoding_maps_ = {}

        for entity, entity_keys in keys.items():
            statistics = (
                pd.DataFrame({"key": entity_keys, "target": targets})
                .groupby("key", sort=False, observed=True)["target"]
                .agg(["count", "sum"])
            )
            counts = statistics["count"].astype(int).to_dict()
            fraud_counts = statistics["sum"].astype(int).to_dict()
            encodings = (
                (statistics["sum"] + self.alpha * self.global_fraud_rate_)
                / (statistics["count"] + self.alpha)
            ).astype(float)
            self.entity_counts_[entity] = counts
            self.entity_fraud_counts_[entity] = fraud_counts
            self.encoding_maps_[entity] = encodings.to_dict()

        history_columns = sorted(
            {
                "TransactionID",
                "TransactionDT",
                "isFraud",
                *BASE_TARGET_ENTITIES,
                *(
                    column
                    for identity in DEFAULT_COMPOSITE_IDENTITIES
                    for column in identity.columns
                ),
            }
        )
        self._fit_history_ = dataframe.loc[:, history_columns].copy()

        self.is_fitted_ = True
        LOGGER.info(
            "Fitted target history for %d entities using %d labeled rows with %.1f-day delay",
            len(keys),
            len(dataframe),
            self.label_delay_days,
        )
        return self

    def fit_transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return strict temporal-OOF TRAIN features and fit frozen future maps."""
        self._validate_input(dataframe, require_target=True)
        features, statistics = self._delayed_features(dataframe, dataframe)
        self.training_statistics_ = statistics
        self.fit(dataframe)
        return features

    def transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen fit-period maps; targets in the input are ignored."""
        if not self.is_fitted_ or self.global_fraud_rate_ is None or self._fit_history_ is None:
            raise RuntimeError("TemporalTargetEncoder must be fitted before transform")
        self._validate_input(dataframe, require_target=False)
        features, statistics = self._delayed_features(dataframe, self._fit_history_)
        self.last_transform_statistics_ = statistics
        return features

    def _delayed_features(
        self,
        queries: pd.DataFrame,
        history: pd.DataFrame,
    ) -> tuple[pd.DataFrame, LabelAvailabilityStatistics]:
        """Encode queries from history labels that have legally matured by query time."""
        query_working = queries.reset_index(drop=True).copy()
        query_working["__position"] = np.arange(len(query_working))
        query_working = query_working.sort_values(
            ["TransactionDT", "TransactionID"], kind="mergesort", ignore_index=True
        )
        history_working = history.sort_values(
            ["TransactionDT", "TransactionID"], kind="mergesort", ignore_index=True
        )
        query_times = pd.to_numeric(query_working["TransactionDT"], errors="raise").to_numpy(
            dtype=np.float64
        )
        history_times = pd.to_numeric(history_working["TransactionDT"], errors="raise").to_numpy(
            dtype=np.float64
        )
        history_targets = self._targets(history_working).to_numpy(dtype=np.int64)
        cutoffs = query_times - self.label_delay_seconds
        if self.label_delay_seconds == 0:
            cutoffs = np.nextafter(cutoffs, -np.inf)

        global_count, global_fraud = self._global_matured_state(
            cutoffs, history_times, history_targets
        )
        global_priors = np.divide(
            global_fraud,
            global_count,
            out=np.full(len(queries), self.initial_prior, dtype=np.float64),
            where=global_count > 0,
        )
        query_keys = self._entity_keys(query_working)
        history_keys = self._entity_keys(history_working)
        features = pd.DataFrame(index=query_working.index)
        supports: dict[str, float] = {}

        for entity, entity_query_keys in query_keys.items():
            entity_count, entity_fraud = self._entity_matured_state(
                entity_query_keys,
                cutoffs,
                history_keys[entity],
                history_times,
                history_targets,
            )
            encoded = (entity_fraud + self.alpha * global_priors) / (entity_count + self.alpha)
            features[f"{entity}_target_encoded_fraud_rate"] = encoded.astype("float32")
            supports[entity] = float(entity_count.mean())

        features["__position"] = query_working["__position"]
        features = (
            features.sort_values("__position")
            .drop(columns="__position")
            .reset_index(drop=True)
            .set_axis(queries.index)
            .loc[:, list(self.feature_names_)]
        )
        representative_positions = np.asarray([0, len(queries) // 2, len(queries) - 1])
        matured_fractions = global_count / len(history)
        fallback_count = int((global_count == 0).sum())
        statistics = LabelAvailabilityStatistics(
            row_count=len(queries),
            history_row_count=len(history),
            fallback_row_count=fallback_count,
            fallback_fraction=fallback_count / len(queries),
            mean_matured_label_count=float(global_count.mean()),
            mean_matured_label_fraction=float(matured_fractions.mean()),
            representative_matured_fractions=tuple(
                float(value) for value in matured_fractions[representative_positions]
            ),
            average_entity_support=supports,
        )
        return features, statistics

    @staticmethod
    def _global_matured_state(
        cutoffs: np.ndarray,
        history_times: np.ndarray,
        history_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batches = (
            pd.DataFrame({"time": history_times, "target": history_targets})
            .groupby("time", sort=True, observed=True)["target"]
            .agg(["count", "sum"])
            .reset_index()
        )
        counts = batches["count"].cumsum().to_numpy(dtype=np.int64)
        fraud = batches["sum"].cumsum().to_numpy(dtype=np.int64)
        positions = np.searchsorted(batches["time"].to_numpy(), cutoffs, side="right") - 1
        matured_count = np.where(positions >= 0, counts[np.maximum(positions, 0)], 0)
        matured_fraud = np.where(positions >= 0, fraud[np.maximum(positions, 0)], 0)
        return matured_count, matured_fraud

    @staticmethod
    def _entity_matured_state(
        query_keys: pd.Series,
        cutoffs: np.ndarray,
        history_keys: pd.Series,
        history_times: np.ndarray,
        history_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        history_batches = (
            pd.DataFrame(
                {
                    "key": history_keys.to_numpy(),
                    "history_time": history_times,
                    "target": history_targets,
                }
            )
            .groupby(["key", "history_time"], sort=True, observed=True)["target"]
            .agg(["count", "sum"])
            .reset_index()
            .sort_values(["history_time", "key"], kind="mergesort")
        )
        grouped = history_batches.groupby("key", sort=False, observed=True)
        history_batches["matured_count"] = grouped["count"].cumsum()
        history_batches["matured_fraud"] = grouped["sum"].cumsum()
        query = pd.DataFrame(
            {
                "key": query_keys.to_numpy(),
                "cutoff": cutoffs,
                "__position": np.arange(len(query_keys)),
            }
        ).sort_values(["cutoff", "key"], kind="mergesort")
        matched = pd.merge_asof(
            query,
            history_batches,
            left_on="cutoff",
            right_on="history_time",
            by="key",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("__position")
        counts = matched["matured_count"].fillna(0).to_numpy(dtype=np.int64)
        fraud = matched["matured_fraud"].fillna(0).to_numpy(dtype=np.int64)
        return counts, fraud

    def _entity_keys(self, dataframe: pd.DataFrame) -> dict[str, pd.Series]:
        keys = {
            entity: dataframe[entity].astype("string").fillna(MISSING_ENTITY)
            for entity in BASE_TARGET_ENTITIES
        }
        composite_keys = create_composite_keys(dataframe, DEFAULT_COMPOSITE_IDENTITIES)
        keys.update({entity: composite_keys[entity] for entity in COMPOSITE_TARGET_ENTITIES})
        return keys

    @staticmethod
    def _targets(dataframe: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(dataframe["isFraud"], errors="raise").astype("int8")

    @staticmethod
    def _validate_input(dataframe: pd.DataFrame, *, require_target: bool) -> None:
        if dataframe.empty:
            raise ValueError("Target-encoding input must not be empty")
        required = {"TransactionID", "TransactionDT", *BASE_TARGET_ENTITIES}
        required.update(
            column for identity in DEFAULT_COMPOSITE_IDENTITIES for column in identity.columns
        )
        if require_target:
            required.add("isFraud")
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise ValueError(f"Target-encoding input is missing required columns: {missing}")
        if dataframe["TransactionID"].duplicated().any():
            raise ValueError("TransactionID must be unique for temporal target encoding")
        times = pd.to_numeric(dataframe["TransactionDT"], errors="raise")
        if times.isna().any() or not np.isfinite(times).all():
            raise ValueError("TransactionDT must contain finite, non-missing values")
        if require_target:
            targets = pd.to_numeric(dataframe["isFraud"], errors="raise")
            if targets.isna().any() or not set(targets.unique()).issubset({0, 1}):
                raise ValueError("isFraud must contain only 0 and 1")
