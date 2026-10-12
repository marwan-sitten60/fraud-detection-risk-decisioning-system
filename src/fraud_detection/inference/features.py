"""Persistent orchestration of the already-frozen feature components."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud_detection.features import (
    COMPOSITE_TARGET_ENTITIES,
    BehavioralFeatureEngineer,
    CompositeFeatureEngineer,
    Phase1FeatureEngineer,
    TemporalTargetEncoder,
)
from fraud_detection.models.comparison import TrainingOnlyPreprocessor

BEHAVIOR_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "DeviceInfo",
    "P_emaildomain",
)


@dataclass(slots=True)
class FrozenFeatureState:
    """TRAIN-fitted maps and histories, separate from any runtime updates."""

    raw_columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    frequency_engineer: Phase1FeatureEngineer
    amount_engineer: BehavioralFeatureEngineer
    composite_engineer: CompositeFeatureEngineer
    target_encoder: TemporalTargetEncoder
    preprocessor: TrainingOnlyPreprocessor
    scaler: StandardScaler

    @classmethod
    def fit_train(cls, train: pd.DataFrame) -> tuple["FrozenFeatureState", np.ndarray]:
        """Materialize the exact notebook-18 TRAIN feature path and retain state."""
        raw_columns = tuple(column for column in train.columns if column != "isFraud")
        predictor_columns = [
            column for column in train.columns if column not in ("isFraud", "TransactionID")
        ]
        train_predictors = train.loc[:, predictor_columns]
        frequency_engineer = Phase1FeatureEngineer(
            include_amount_time=False,
            include_missing_indicators=False,
            include_frequency_encodings=True,
        ).fit(train_predictors)
        frequency_train = frequency_engineer.transform(train_predictors)
        behavior_columns = [column for column in BEHAVIOR_COLUMNS if column in train.columns]
        amount_engineer = BehavioralFeatureEngineer(
            include_counts=False,
            include_amounts=True,
            include_recency=False,
            include_velocity=False,
        )
        amount_train = amount_engineer.fit_transform_features(train.loc[:, behavior_columns])
        engineered_train = pd.concat([frequency_train, amount_train], axis=1)
        composite_engineer = CompositeFeatureEngineer()
        composite_train = composite_engineer.fit_transform_features(train)
        engineered_train = pd.concat([engineered_train, composite_train], axis=1)
        target_encoder = TemporalTargetEncoder(alpha=50.0, initial_prior=0.5, label_delay_days=0.0)
        target_train = target_encoder.fit_transform_features(train)
        engineered_train = pd.concat([engineered_train, target_train], axis=1)
        preprocessor = TrainingOnlyPreprocessor()
        x_train = preprocessor.fit_transform(engineered_train)
        scaler = StandardScaler(copy=False)
        scaler.partial_fit(x_train)
        x_train = scaler.transform(x_train)
        feature_names = _selected_feature_names(tuple(preprocessor.feature_names_))
        positions = {name: index for index, name in enumerate(preprocessor.feature_names_)}
        selected = x_train[:, [positions[name] for name in feature_names]]
        state = cls(
            raw_columns,
            feature_names,
            frequency_engineer,
            amount_engineer,
            composite_engineer,
            target_encoder,
            preprocessor,
            scaler,
        )
        return state, selected

    def transform(self, rows: pd.DataFrame, *, update_state: bool = True) -> np.ndarray:
        """Transform ordered label-free rows with frozen maps and optional history updates."""
        predictor_columns = [column for column in self.raw_columns if column != "TransactionID"]
        frequency = self.frequency_engineer.transform(rows.loc[:, predictor_columns])
        behavior_columns = [column for column in BEHAVIOR_COLUMNS if column in rows.columns]
        amounts = self.amount_engineer.transform_features(
            rows.loc[:, behavior_columns], update_state=update_state
        )
        composite = self.composite_engineer.transform_features(rows, update_state=update_state)
        target = self.target_encoder.transform_features(rows)
        engineered = pd.concat([frequency, amounts, composite, target], axis=1)
        matrix = self.scaler.transform(self.preprocessor.transform(engineered))
        positions = {name: index for index, name in enumerate(self.preprocessor.feature_names_)}
        return matrix[:, [positions[name] for name in self.feature_names]]


def _selected_feature_names(all_names: tuple[str, ...]) -> tuple[str, ...]:
    baseline = tuple(name for name in all_names if not name.endswith("_target_encoded_fraud_rate"))
    composite_target = tuple(
        f"{entity}_target_encoded_fraud_rate" for entity in COMPOSITE_TARGET_ENTITIES
    )
    return baseline + composite_target
