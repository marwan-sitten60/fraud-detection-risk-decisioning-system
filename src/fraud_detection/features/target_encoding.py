"""Leakage-safe temporal target encoding for fraud-risk entities."""

import logging

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


class TemporalTargetEncoder:
    """Create strict temporal-OOF encodings and frozen future-period maps.

    Training rows at timestamp ``T`` use labels only from timestamps strictly
    before ``T``. The earliest timestamp uses ``initial_prior`` because no legal
    historical label exists. ``fit`` stores plain dictionaries and scalars so
    the frozen state remains straightforward to serialize for production.
    """

    def __init__(self, *, alpha: float = 50.0, initial_prior: float = 0.5) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a positive finite number")
        if not np.isfinite(initial_prior) or not 0 <= initial_prior <= 1:
            raise ValueError("initial_prior must be between 0 and 1")
        self.alpha = float(alpha)
        self.initial_prior = float(initial_prior)
        self.feature_names_ = TARGET_ENCODING_FEATURES
        self.global_fraud_rate_: float | None = None
        self.entity_counts_: dict[str, dict[str, int]] = {}
        self.entity_fraud_counts_: dict[str, dict[str, int]] = {}
        self.encoding_maps_: dict[str, dict[str, float]] = {}
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

        self.is_fitted_ = True
        LOGGER.info(
            "Fitted frozen temporal target maps for %d entities using %d labeled rows",
            len(keys),
            len(dataframe),
        )
        return self

    def fit_transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return strict temporal-OOF TRAIN features and fit frozen future maps."""
        self._validate_input(dataframe, require_target=True)
        features = self._temporal_oof_features(dataframe)
        self.fit(dataframe)
        return features

    def transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen fit-period maps; targets in the input are ignored."""
        if not self.is_fitted_ or self.global_fraud_rate_ is None:
            raise RuntimeError("TemporalTargetEncoder must be fitted before transform")
        self._validate_input(dataframe, require_target=False)
        keys = self._entity_keys(dataframe)
        features = pd.DataFrame(index=dataframe.index)
        for entity, entity_keys in keys.items():
            feature_name = f"{entity}_target_encoded_fraud_rate"
            features[feature_name] = (
                entity_keys.map(self.encoding_maps_[entity])
                .fillna(self.global_fraud_rate_)
                .astype("float32")
            )
        return features.loc[:, list(self.feature_names_)]

    def _temporal_oof_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        working = dataframe.reset_index(drop=True).copy()
        working["__position"] = np.arange(len(working))
        working = working.sort_values(
            ["TransactionDT", "TransactionID"], kind="mergesort", ignore_index=True
        )
        targets = self._targets(working)
        times = pd.to_numeric(working["TransactionDT"], errors="raise")
        keys = self._entity_keys(working)
        global_priors = self._historical_global_priors(times, targets)
        features = pd.DataFrame(index=working.index)

        for entity, entity_keys in keys.items():
            batches = (
                pd.DataFrame({"key": entity_keys, "time": times, "target": targets})
                .groupby(["key", "time"], sort=False, observed=True)["target"]
                .agg(["count", "sum"])
                .reset_index()
            )
            grouped = batches.groupby("key", sort=False, observed=True)
            batches["previous_count"] = grouped["count"].cumsum() - batches["count"]
            batches["previous_fraud"] = grouped["sum"].cumsum() - batches["sum"]
            lookup = batches.set_index(["key", "time"])
            row_index = pd.MultiIndex.from_arrays([entity_keys, times])
            previous_count = lookup["previous_count"].reindex(row_index).to_numpy()
            previous_fraud = lookup["previous_fraud"].reindex(row_index).to_numpy()
            encoded = (previous_fraud + self.alpha * global_priors) / (previous_count + self.alpha)
            features[f"{entity}_target_encoded_fraud_rate"] = encoded.astype("float32")

        features["__position"] = working["__position"]
        return (
            features.sort_values("__position")
            .drop(columns="__position")
            .reset_index(drop=True)
            .set_axis(dataframe.index)
            .loc[:, list(self.feature_names_)]
        )

    def _historical_global_priors(
        self,
        times: pd.Series,
        targets: pd.Series,
    ) -> np.ndarray:
        batches = (
            pd.DataFrame({"time": times, "target": targets})
            .groupby("time", sort=False, observed=True)["target"]
            .agg(["count", "sum"])
            .reset_index()
        )
        batches["previous_count"] = batches["count"].cumsum() - batches["count"]
        batches["previous_fraud"] = batches["sum"].cumsum() - batches["sum"]
        batches["prior"] = batches["previous_fraud"].div(
            batches["previous_count"].where(batches["previous_count"] > 0)
        )
        batches["prior"] = batches["prior"].fillna(self.initial_prior)
        return batches.set_index("time")["prior"].reindex(times).to_numpy(dtype=np.float64)

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
