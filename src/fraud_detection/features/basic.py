"""Safe, interpretable Phase 1 features for tabular fraud models."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_MISSING_INDICATOR_COLUMNS = (
    "DeviceType",
    "DeviceInfo",
    "R_emaildomain",
    "P_emaildomain",
    "dist1",
)
DEFAULT_FREQUENCY_COLUMNS = (
    "card1",
    "card2",
    "addr1",
    "DeviceInfo",
    "P_emaildomain",
    "R_emaildomain",
)
MISSING_CATEGORY = "<MISSING>"


class Phase1FeatureEngineer:
    """Fit and apply target-independent Phase 1 features.

    ``TransactionDT`` is an origin-unknown relative counter in seconds. The fitted
    training minimum defines elapsed day and repeating relative hour; no calendar
    date or timezone is inferred.
    """

    def __init__(
        self,
        *,
        include_amount_time: bool = True,
        include_missing_indicators: bool = True,
        include_frequency_encodings: bool = True,
        missing_indicator_columns: Iterable[str] = DEFAULT_MISSING_INDICATOR_COLUMNS,
        frequency_columns: Iterable[str] = DEFAULT_FREQUENCY_COLUMNS,
    ) -> None:
        self.include_amount_time = include_amount_time
        self.include_missing_indicators = include_missing_indicators
        self.include_frequency_encodings = include_frequency_encodings
        self.missing_indicator_columns = tuple(missing_indicator_columns)
        self.frequency_columns = tuple(frequency_columns)
        self.transaction_dt_origin_: float | None = None
        self.frequency_maps_: dict[str, dict[str, float]] = {}
        self.generated_feature_names_: tuple[str, ...] = ()
        self.is_fitted_ = False

    def fit(self, dataframe: pd.DataFrame) -> "Phase1FeatureEngineer":
        """Learn relative-time origin and category frequencies from training only."""
        generated_names: list[str] = []
        if self.include_amount_time:
            self._require_columns(dataframe, ("TransactionAmt", "TransactionDT"))
            self._validate_amounts(dataframe["TransactionAmt"])
            transaction_dt = pd.to_numeric(dataframe["TransactionDT"], errors="raise")
            if transaction_dt.isna().any():
                raise ValueError("TransactionDT must not contain missing values")
            self.transaction_dt_origin_ = float(transaction_dt.min())
            generated_names.extend(("log_transaction_amount", "elapsed_day", "relative_hour"))

        if self.include_missing_indicators:
            generated_names.extend(
                f"{column}_is_missing"
                for column in self.missing_indicator_columns
                if column in dataframe.columns
            )

        self.frequency_maps_ = {}
        if self.include_frequency_encodings:
            for column in self.frequency_columns:
                if column not in dataframe.columns:
                    continue
                normalized = self._normalize_categories(dataframe[column])
                self.frequency_maps_[column] = (
                    normalized.value_counts(normalize=True).astype(float).to_dict()
                )
                generated_names.append(f"{column}_frequency")

        self.generated_feature_names_ = tuple(generated_names)
        self.is_fitted_ = True
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with fitted Phase 1 features appended."""
        if not self.is_fitted_:
            raise RuntimeError("Phase1FeatureEngineer must be fitted before transform")

        transformed = dataframe.copy()
        if self.include_amount_time:
            self._require_columns(transformed, ("TransactionAmt", "TransactionDT"))
            self._validate_amounts(transformed["TransactionAmt"])
            transaction_dt = pd.to_numeric(transformed["TransactionDT"], errors="raise")
            if transaction_dt.isna().any():
                raise ValueError("TransactionDT must not contain missing values")
            if self.transaction_dt_origin_ is None:
                raise RuntimeError("Training TransactionDT origin is unavailable")
            relative_seconds = transaction_dt - self.transaction_dt_origin_
            transformed["log_transaction_amount"] = np.log1p(transformed["TransactionAmt"])
            transformed["elapsed_day"] = np.floor_divide(relative_seconds, 86_400).astype("int32")
            transformed["relative_hour"] = (
                np.floor_divide(relative_seconds, 3_600).astype("int32") % 24
            )

        if self.include_missing_indicators:
            for column in self.missing_indicator_columns:
                if column in transformed.columns:
                    transformed[f"{column}_is_missing"] = transformed[column].isna().astype("int8")

        if self.include_frequency_encodings:
            for column, frequency_map in self.frequency_maps_.items():
                if column not in transformed.columns:
                    transformed[f"{column}_frequency"] = np.float32(0.0)
                    continue
                normalized = self._normalize_categories(transformed[column])
                transformed[f"{column}_frequency"] = (
                    normalized.map(frequency_map).fillna(0.0).astype("float32")
                )
        return transformed

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return its engineered copy."""
        return self.fit(dataframe).transform(dataframe)

    @staticmethod
    def _normalize_categories(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna(MISSING_CATEGORY)

    @staticmethod
    def _require_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> None:
        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Phase 1 feature engineering requires columns: {missing}")

    @staticmethod
    def _validate_amounts(amounts: pd.Series) -> None:
        numeric_amounts = pd.to_numeric(amounts, errors="raise")
        if (numeric_amounts.dropna() < 0).any():
            raise ValueError("TransactionAmt must be non-negative for log1p transformation")
