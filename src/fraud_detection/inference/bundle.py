"""Versioned, checksummed artifacts for the frozen inference runtime."""

import hashlib
import json
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import joblib
import lightgbm as lgb
import xgboost as xgb

from fraud_detection.decisioning.costs import scenario_profiles
from fraud_detection.inference.features import FrozenFeatureState

BUNDLE_VERSION = "frozen_v1"
MANIFEST_NAME = "artifact_manifest.json"
MANIFEST_HASH_NAME = "artifact_manifest.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_feature_hash(names: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()


def write_bundle(
    destination: Path,
    *,
    feature_state: FrozenFeatureState,
    lightgbm_model: lgb.LGBMClassifier,
    xgboost_model: xgb.XGBClassifier,
    calibrator_path: Path,
    rules_path: Path,
    lightgbm_config: dict[str, object],
    xgboost_config: dict[str, object],
    scale_pos_weight: float,
    frozen_manifest_path: Path,
    holdout_report_path: Path,
) -> dict[str, object]:
    """Persist the TRAIN-fitted system once, with native booster formats."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Frozen bundle already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_state, destination / "feature_state.joblib", compress=3)
    lightgbm_model.booster_.save_model(str(destination / "lightgbm.txt"))
    xgboost_model.get_booster().save_model(str(destination / "xgboost.json"))
    shutil.copyfile(calibrator_path, destination / "platt_calibrator.joblib")
    shutil.copyfile(rules_path, destination / "rules_frozen_config.json")
    policy = {
        "primary_policy": "Balanced expected-cost, rules disabled",
        "rules_enabled_by_default": False,
        "ensemble_weights": {"lightgbm": 0.8, "xgboost": 0.2},
        "target_encoding_label_delay_days": 0.0,
        "fit_scope": "TRAIN only",
        "balanced_cost_config": asdict(scenario_profiles()["Balanced"]),
    }
    (destination / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    schema = {
        "raw_columns": list(feature_state.raw_columns),
        "feature_names": list(feature_state.feature_names),
    }
    (destination / "feature_contract.json").write_text(json.dumps(schema, indent=2) + "\n")
    artifact_files = {
        "feature_state": "feature_state.joblib",
        "lightgbm": "lightgbm.txt",
        "xgboost": "xgboost.json",
        "calibrator": "platt_calibrator.joblib",
        "rules": "rules_frozen_config.json",
        "policy": "policy.json",
        "feature_contract": "feature_contract.json",
    }
    serializers = {
        "feature_state": "joblib (pandas/sklearn/project state)",
        "lightgbm": "LightGBM native text",
        "xgboost": "XGBoost native JSON",
        "calibrator": "existing joblib artifact",
        "rules": "JSON, unchanged copy",
        "policy": "JSON",
        "feature_contract": "JSON",
    }
    packages = {
        name: version(name)
        for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "joblib")
    }
    manifest: dict[str, object] = {
        "bundle_version": BUNDLE_VERSION,
        "creation_purpose": "Artifact materialization of the terminal evaluated TRAIN-only system",
        "fit_scope": "TRAIN only",
        "feature_count": len(feature_state.feature_names),
        "ordered_feature_names_sha256": ordered_feature_hash(feature_state.feature_names),
        "lightgbm_configuration": lightgbm_config,
        "xgboost_configuration": {**xgboost_config, "scale_pos_weight": scale_pos_weight},
        "ensemble_weights": policy["ensemble_weights"],
        "calibration_method": "platt",
        "decision_policy_identifier": "Balanced expected-cost",
        "rules_enabled_by_default": False,
        "software_versions": packages,
        "frozen_system_manifest": str(frozen_manifest_path),
        "terminal_holdout_evaluation": str(holdout_report_path),
        "artifact_files": artifact_files,
        "serializers": serializers,
        "sha256": {
            name: sha256_file(destination / filename) for name, filename in artifact_files.items()
        },
    }
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / MANIFEST_HASH_NAME).write_text(sha256_file(manifest_path) + "\n")
    return manifest


def verify_bundle(destination: Path) -> dict[str, object]:
    """Validate all payload checksums before deserializing any artifact."""
    manifest_path = destination / MANIFEST_NAME
    expected_manifest_hash = (destination / MANIFEST_HASH_NAME).read_text().strip()
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError("Bundle manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["bundle_version"] != BUNDLE_VERSION:
        raise ValueError(f"Unsupported bundle version: {manifest['bundle_version']}")
    for name, filename in manifest["artifact_files"].items():
        if sha256_file(destination / filename) != manifest["sha256"][name]:
            raise ValueError(f"Bundle checksum mismatch: {filename}")
    return manifest
