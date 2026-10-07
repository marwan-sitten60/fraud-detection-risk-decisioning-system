"""Leakage-safe composite identity features for chronological fraud data."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MISSING_COMPONENT = "M"


@dataclass(frozen=True, slots=True)
class CompositeIdentity:
    """A named, deterministic combination of existing transaction fields."""

    name: str
    columns: tuple[str, ...]


DEFAULT_COMPOSITE_IDENTITIES = (
    CompositeIdentity("card1_card2_addr1", ("card1", "card2", "addr1")),
    CompositeIdentity(
        "card1_addr1_p_emaildomain",
        ("card1", "addr1", "P_emaildomain"),
    ),
    CompositeIdentity(
        "card1_card2_addr1_productcd",
        ("card1", "card2", "addr1", "ProductCD"),
    ),
    CompositeIdentity(
        "card1_addr1_deviceinfo",
        ("card1", "addr1", "DeviceInfo"),
    ),
)


def create_composite_keys(
    dataframe: pd.DataFrame,
    identities: tuple[CompositeIdentity, ...] = DEFAULT_COMPOSITE_IDENTITIES,
) -> pd.DataFrame:
    """Create unambiguous composite keys without mutating the source dataframe.

    Missing components use the explicit token ``M``. Nonmissing values use
    ``V<length>:<value>`` before components are joined, preventing delimiter and
    missing-token collisions.
    """
    required = {column for identity in identities for column in identity.columns}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Composite identity input is missing required columns: {missing}")

    keys: dict[str, pd.Series] = {}
    for identity in identities:
        components = [_encode_component(dataframe[column]) for column in identity.columns]
        key = components[0]
        for component in components[1:]:
            key = key + "|" + component
        keys[identity.name] = key
    return pd.DataFrame(keys, index=dataframe.index)


class CompositeFeatureEngineer:
    """Generate fold-local frequency and strict past-only composite history.

    Historical features use only rows with ``TransactionDT < T``. All rows at
    the same timestamp receive features from the earlier state, and the complete
    timestamp batch updates history only afterward. Static frequency maps are
    fitted on the training/reference period and unseen identities map to zero.
    """

    def __init__(
        self,
        *,
        include_frequency: bool = True,
        include_history: bool = True,
        identities: tuple[CompositeIdentity, ...] = DEFAULT_COMPOSITE_IDENTITIES,
    ) -> None:
        if not include_frequency and not include_history:
            raise ValueError("At least one composite feature group must be enabled")
        if not identities:
            raise ValueError("At least one composite identity is required")
        self.include_frequency = include_frequency
        self.include_history = include_history
        self.identities = identities
        self.frequency_maps_: dict[str, dict[str, float]] = {}
        self.feature_groups_: dict[str, tuple[str, ...]] = self._feature_groups()
        self.generated_feature_names_ = tuple(
            feature for group in self.feature_groups_.values() for feature in group
        )
        self.history_: pd.DataFrame | None = None
        self.is_fitted_ = False

    def fit(self, dataframe: pd.DataFrame) -> "CompositeFeatureEngineer":
        """Fit reference frequencies and initialize target-independent history."""
        self._validate_input(dataframe)
        keys = create_composite_keys(dataframe, self.identities)
        self._fit_frequency_maps(keys)
        self.history_ = self._history_columns(dataframe).copy()
        self.is_fitted_ = True
        return self

    def fit_transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit on a reference period and return its composite numeric features."""
        self._validate_input(dataframe)
        keys = create_composite_keys(dataframe, self.identities)
        self._fit_frequency_maps(keys)
        features = self._frequency_features(keys)
        if self.include_history:
            features = pd.concat([features, self._historical_features(dataframe)], axis=1)
        self.history_ = self._history_columns(dataframe).copy()
        self.is_fitted_ = True
        LOGGER.info(
            "Generated %d composite features for %d reference transactions",
            features.shape[1],
            len(dataframe),
        )
        return features.loc[:, list(self.generated_feature_names_)]

    def transform_features(
        self,
        dataframe: pd.DataFrame,
        *,
        update_state: bool = True,
    ) -> pd.DataFrame:
        """Transform a later period using fitted frequencies and prior history."""
        if not self.is_fitted_ or self.history_ is None:
            raise RuntimeError("CompositeFeatureEngineer must be fitted before transform")
        self._validate_input(dataframe)
        self._validate_period_follows_history(dataframe)

        keys = create_composite_keys(dataframe, self.identities)
        features = self._frequency_features(keys)
        if self.include_history:
            combined = pd.concat(
                [self.history_, self._history_columns(dataframe)],
                ignore_index=True,
                copy=False,
            )
            combined_features = self._historical_features(combined)
            current_history = combined_features.iloc[-len(dataframe) :].copy()
            current_history.index = dataframe.index
            features = pd.concat([features, current_history], axis=1)
        if update_state:
            self.history_ = pd.concat(
                [self.history_, self._history_columns(dataframe)],
                ignore_index=True,
                copy=False,
            )
        LOGGER.info(
            "Generated composite features for %d sequential transactions; update_state=%s",
            len(dataframe),
            update_state,
        )
        return features.loc[:, list(self.generated_feature_names_)]

    def _fit_frequency_maps(self, keys: pd.DataFrame) -> None:
        self.frequency_maps_ = {}
        if not self.include_frequency:
            return
        for identity in self.identities:
            self.frequency_maps_[identity.name] = (
                keys[identity.name].value_counts(normalize=True).astype(float).to_dict()
            )

    def _frequency_features(self, keys: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=keys.index)
        if not self.include_frequency:
            return features
        for identity in self.identities:
            feature_name = f"{identity.name}_composite_frequency"
            features[feature_name] = (
                keys[identity.name]
                .map(self.frequency_maps_[identity.name])
                .fillna(0.0)
                .astype("float32")
            )
        return features

    def _historical_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        working = self._history_columns(dataframe).reset_index(drop=True)
        working["__position"] = np.arange(len(working))
        working = working.sort_values(
            ["TransactionDT", "TransactionID"], kind="mergesort", ignore_index=True
        )
        keys = create_composite_keys(working, self.identities)
        times = pd.to_numeric(working["TransactionDT"], errors="raise")
        amounts = pd.to_numeric(working["TransactionAmt"], errors="raise").astype("float64")
        features = pd.DataFrame(index=working.index)

        for identity in self.identities:
            identity_keys = keys[identity.name]
            batches = (
                pd.DataFrame(
                    {
                        "__key": identity_keys,
                        "TransactionDT": times,
                        "TransactionAmt": amounts,
                    }
                )
                .groupby(["__key", "TransactionDT"], sort=False, observed=True)
                .agg(
                    batch_count=("TransactionAmt", "size"),
                    batch_sum=("TransactionAmt", "sum"),
                    batch_max=("TransactionAmt", "max"),
                )
                .reset_index()
            )
            grouped = batches.groupby("__key", sort=False, observed=True)
            batches["previous_count"] = grouped["batch_count"].cumsum() - batches["batch_count"]
            batches["previous_sum"] = grouped["batch_sum"].cumsum() - batches["batch_sum"]
            cumulative_max = grouped["batch_max"].cummax()
            batches["previous_max"] = cumulative_max.groupby(batches["__key"], sort=False).shift(1)
            batches["previous_avg"] = batches["previous_sum"].div(
                batches["previous_count"].where(batches["previous_count"] > 0)
            )

            lookup = batches.set_index(["__key", "TransactionDT"])
            row_index = pd.MultiIndex.from_arrays([identity_keys, times])
            previous_count = lookup["previous_count"].reindex(row_index).to_numpy()
            previous_avg = lookup["previous_avg"].reindex(row_index).to_numpy()
            previous_max = lookup["previous_max"].reindex(row_index).to_numpy()
            prefix = identity.name
            features[f"{prefix}_previous_tx_count"] = previous_count.astype("int32")
            features[f"{prefix}_previous_avg_amount"] = previous_avg.astype("float32")
            features[f"{prefix}_previous_max_amount"] = previous_max.astype("float32")
            features[f"{prefix}_amount_vs_previous_avg"] = _safe_ratio(
                amounts.to_numpy(), previous_avg
            )

        features["__position"] = working["__position"]
        return (
            features.sort_values("__position")
            .drop(columns="__position")
            .reset_index(drop=True)
            .set_axis(dataframe.index)
        )

    def _feature_groups(self) -> dict[str, tuple[str, ...]]:
        frequency = tuple(f"{identity.name}_composite_frequency" for identity in self.identities)
        history = tuple(
            feature
            for identity in self.identities
            for feature in (
                f"{identity.name}_previous_tx_count",
                f"{identity.name}_previous_avg_amount",
                f"{identity.name}_previous_max_amount",
                f"{identity.name}_amount_vs_previous_avg",
            )
        )
        return {
            "composite_frequency": frequency if self.include_frequency else (),
            "composite_history": history if self.include_history else (),
        }

    def _history_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        columns = ["TransactionID", "TransactionDT", "TransactionAmt"]
        columns.extend(
            dict.fromkeys(column for identity in self.identities for column in identity.columns)
        )
        return dataframe.loc[:, columns]

    def _validate_input(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            raise ValueError("Composite feature input must not be empty")
        required = {"TransactionID", "TransactionDT", "TransactionAmt"}
        required.update(column for identity in self.identities for column in identity.columns)
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise ValueError(f"Composite feature input is missing required columns: {missing}")
        if dataframe["TransactionID"].duplicated().any():
            raise ValueError("TransactionID must be unique for deterministic composite history")
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
        if dataframe["TransactionDT"].min() < self.history_["TransactionDT"].max():
            raise ValueError("New transactions must not precede fitted composite history")


def _encode_component(series: pd.Series) -> pd.Series:
    normalized = series.astype("string")
    encoded = "V" + normalized.str.len().astype("Int64").astype("string") + ":" + normalized
    return encoded.fillna(MISSING_COMPONENT)


def _safe_ratio(amounts: np.ndarray, previous_average: np.ndarray) -> np.ndarray:
    result = np.full(len(amounts), np.nan, dtype=np.float32)
    valid = np.isfinite(previous_average) & (previous_average != 0)
    result[valid] = (amounts[valid] / previous_average[valid]).astype(np.float32)
    return result
