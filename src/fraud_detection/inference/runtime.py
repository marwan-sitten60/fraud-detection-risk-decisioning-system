"""Label-free scoring from the immutable frozen artifact bundle."""

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_detection.decisioning.costs import BusinessCostConfig
from fraud_detection.decisioning.policy import decide
from fraud_detection.decisioning.rules import Rule, RulesEngine, RuleThresholds
from fraud_detection.features.composite import CompositeIdentity, create_composite_keys
from fraud_detection.inference.bundle import ordered_feature_hash, verify_bundle
from fraud_detection.inference.contracts import validate_transactions
from fraud_detection.inference.features import FrozenFeatureState

RULE_COMPOSITE = (
    CompositeIdentity("card1_addr1_p_emaildomain", ("card1", "addr1", "P_emaildomain")),
)


class FrozenFraudInferenceRuntime:
    """A loaded bundle plus instance-local, chronological non-label history."""

    def __init__(
        self,
        bundle_path: Path,
        manifest: dict[str, object],
        feature_state: FrozenFeatureState,
        lightgbm: lgb.Booster,
        xgboost: xgb.Booster,
        calibrator: object,
        costs: BusinessCostConfig,
        rules: RulesEngine,
    ) -> None:
        self.bundle_path = bundle_path
        self.manifest = manifest
        self.feature_state = feature_state
        self.lightgbm = lightgbm
        self.xgboost = xgboost
        self.calibrator = calibrator
        self.costs = costs
        self.rules = rules
        self.last_timings_: dict[str, float] = {}

    @classmethod
    def load(cls, bundle_path: str | Path) -> "FrozenFraudInferenceRuntime":
        """Verify checksums before loading any serialized fitted state."""
        destination = Path(bundle_path)
        manifest = verify_bundle(destination)
        files = manifest["artifact_files"]
        schema = json.loads((destination / files["feature_contract"]).read_text())
        policy = json.loads((destination / files["policy"]).read_text())
        state = joblib.load(destination / files["feature_state"])
        if not isinstance(state, FrozenFeatureState):
            raise ValueError("Bundle feature state has an unexpected type")
        if tuple(schema["raw_columns"]) != state.raw_columns:
            raise ValueError("Bundle raw-column contract mismatch")
        if tuple(schema["feature_names"]) != state.feature_names:
            raise ValueError("Bundle ordered feature names mismatch")
        if len(state.feature_names) != 467 or manifest["feature_count"] != 467:
            raise ValueError("Frozen feature count must equal 467")
        if ordered_feature_hash(state.feature_names) != manifest["ordered_feature_names_sha256"]:
            raise ValueError("Bundle ordered feature hash mismatch")
        if policy["rules_enabled_by_default"] is not False:
            raise ValueError("Primary frozen policy must have rules disabled")
        if policy["ensemble_weights"] != {"lightgbm": 0.8, "xgboost": 0.2}:
            raise ValueError("Frozen ensemble weights mismatch")
        calibrator = joblib.load(destination / files["calibrator"])
        if calibrator.method != "platt":
            raise ValueError("Frozen calibration method must be Platt")
        rule_config = json.loads((destination / files["rules"]).read_text())
        rules = RulesEngine(
            RuleThresholds(**rule_config["thresholds"]),
            tuple(Rule(**item) for item in rule_config["rules"]),
        )
        return cls(
            destination,
            manifest,
            state,
            lgb.Booster(model_file=str(destination / files["lightgbm"])),
            xgb.Booster(model_file=str(destination / files["xgboost"])),
            calibrator,
            BusinessCostConfig(**policy["balanced_cost_config"]),
            rules,
        )

    def predict_batch(
        self,
        rows: pd.DataFrame | Sequence[Mapping[str, object]],
        *,
        include_rules: bool = False,
        update_state: bool = True,
    ) -> pd.DataFrame:
        """Score ordered rows; optionally retain label-free history for later batches."""
        started = time.perf_counter()
        numeric_columns = tuple(
            column
            for column in self.feature_state.raw_columns
            if column not in self.feature_state.preprocessor.categorical_features_
        )
        frame = validate_transactions(rows, self.feature_state.raw_columns, numeric_columns)
        features = self.feature_state.transform(frame, update_state=update_state)
        featured = time.perf_counter()
        lightgbm_score = np.asarray(self.lightgbm.predict(features), dtype=np.float64)
        xgboost_score = np.asarray(self.xgboost.predict(xgb.DMatrix(features)), dtype=np.float64)
        ensemble = 0.8 * lightgbm_score + 0.2 * xgboost_score
        modeled = time.perf_counter()
        calibrated = np.asarray(self.calibrator.transform(ensemble), dtype=np.float64)
        base = decide(
            pd.DataFrame(
                {
                    "TransactionID": frame.TransactionID.to_numpy(),
                    "calibrated_probability": calibrated,
                    "TransactionAmt": frame.TransactionAmt.to_numpy(),
                }
            ),
            self.costs,
        )
        result = pd.DataFrame(
            {
                "TransactionID": frame.TransactionID.to_numpy(),
                "lightgbm_score": lightgbm_score,
                "xgboost_score": xgboost_score,
                "ensemble_score": ensemble,
                "calibrated_probability": calibrated,
                "base_decision": base.decision.to_numpy(),
                "expected_cost_approve": base.expected_cost_approve.to_numpy(),
                "expected_cost_review": base.expected_cost_review.to_numpy(),
                "expected_cost_block": base.expected_cost_block.to_numpy(),
                "bundle_version": self.manifest["bundle_version"],
            },
            index=frame.index,
        )
        if include_rules:
            composite_key = create_composite_keys(frame, RULE_COMPOSITE).iloc[:, 0]
            frequency = self.feature_state.composite_engineer.frequency_maps_[
                "card1_addr1_p_emaildomain"
            ]
            signals = result.loc[
                :, ["TransactionID", "calibrated_probability", "lightgbm_score", "xgboost_score"]
            ].copy()
            signals["TransactionAmt"] = frame.TransactionAmt.to_numpy()
            signals["composite_frequency"] = composite_key.map(frequency).fillna(0).to_numpy()
            signals["base_decision"] = base.decision.to_numpy()
            optional = self.rules.evaluate(signals)
            result["final_decision"] = optional.final_decision.to_numpy()
            result["triggered_rule_ids"] = optional.triggered_rule_ids.map(
                lambda ids: "|".join(ids)
            ).to_numpy()
        self.last_timings_ = {
            "feature_seconds": featured - started,
            "model_seconds": modeled - featured,
            "calibration_decision_seconds": time.perf_counter() - modeled,
            "total_seconds": time.perf_counter() - started,
        }
        return result

    def predict_one(
        self, row: Mapping[str, object], *, include_rules: bool = False, update_state: bool = True
    ) -> dict[str, object]:
        """Score one transaction using the same chronological update path."""
        return (
            self.predict_batch([row], include_rules=include_rules, update_state=update_state)
            .iloc[0]
            .to_dict()
        )

    def snapshot_state(self) -> dict[str, pd.DataFrame]:
        """Copy the instance-local non-label histories for explicit restore."""
        return {
            "amount_history": self.feature_state.amount_engineer.history_.copy(deep=True),
            "composite_history": self.feature_state.composite_engineer.history_.copy(deep=True),
        }

    def restore_state(self, snapshot: dict[str, pd.DataFrame]) -> None:
        """Restore a caller-held snapshot without modifying the bundle."""
        self.feature_state.amount_engineer.history_ = snapshot["amount_history"].copy(deep=True)
        self.feature_state.composite_engineer.history_ = snapshot["composite_history"].copy(
            deep=True
        )

    def reset_state(self) -> None:
        """Discard runtime history updates and reload the immutable TRAIN state."""
        filename = self.manifest["artifact_files"]["feature_state"]
        self.feature_state = joblib.load(self.bundle_path / filename)
