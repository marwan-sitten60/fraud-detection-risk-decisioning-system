"""Materialize the terminal evaluated TRAIN-only system once as native artifacts."""

import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.data import merge_train_data  # noqa: E402
from fraud_detection.inference.bundle import write_bundle  # noqa: E402
from fraud_detection.inference.features import FrozenFeatureState  # noqa: E402

TRAIN_ROWS = int(590540 * 0.70)
LIGHTGBM_CONFIG = {
    "learning_rate": 0.1,
    "n_estimators": 400,
    "num_leaves": 63,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}
XGBOOST_CONFIG = {
    "learning_rate": 0.1,
    "max_depth": 7,
    "n_estimators": 300,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}


def main() -> None:
    started = time.perf_counter()
    destination = ROOT / "models/frozen_v1"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to replace existing frozen bundle: {destination}")
    frozen_calibrator = ROOT / "models/probability_calibrator.joblib"
    frozen_rules = ROOT / "reports/rules_frozen_config.json"
    assert hashlib.sha256(frozen_calibrator.read_bytes()).hexdigest() == (
        "3503bf66161b5152e9de4d638b9e7932d581fc371abd6a45b6969ba6c8134f6e"
    )
    assert hashlib.sha256(frozen_rules.read_bytes()).hexdigest() == (
        "3a4380831e177828c82a1a118259f83e354221fa52205b41d38aa21b58febb94"
    )
    transaction_path = ROOT / "data/raw/train_transaction.csv"
    sample = pd.read_csv(transaction_path, nrows=1000, low_memory=False)
    dtypes = {
        name: ("float64" if pd.api.types.is_numeric_dtype(dtype) else "object")
        for name, dtype in sample.dtypes.items()
    }
    dtypes.update(
        {"TransactionID": "int64", "TransactionDT": "int64", "card1": "int64", "isFraud": "int8"}
    )
    train_transaction = pd.read_csv(transaction_path, nrows=TRAIN_ROWS, dtype=dtypes)
    assert len(train_transaction) == TRAIN_ROWS
    train_max_id = int(train_transaction.TransactionID.iloc[-1])
    identity_path = ROOT / "data/raw/train_identity.csv"
    identity_ids = pd.read_csv(identity_path, usecols=["TransactionID"])
    identity_count = int(identity_ids.TransactionID.le(train_max_id).sum())
    assert identity_ids.TransactionID.iloc[:identity_count].le(train_max_id).all()
    if identity_count < len(identity_ids):
        assert identity_ids.TransactionID.iloc[identity_count] > train_max_id
    del identity_ids
    train_identity = pd.read_csv(identity_path, nrows=identity_count, low_memory=False)
    train = merge_train_data(train_transaction, train_identity)
    del train_transaction, train_identity
    assert len(train) == TRAIN_ROWS and train.isFraud.isin([0, 1]).all()
    print("TRAIN-only merged rows:", len(train), flush=True)
    feature_state, x_train = FrozenFeatureState.fit_train(train)
    assert len(feature_state.feature_names) == 467 and x_train.shape == (TRAIN_ROWS, 467)
    print(
        "Frozen feature matrix:",
        x_train.shape,
        "seconds:",
        time.perf_counter() - started,
        flush=True,
    )
    y_train = train.isFraud.astype("int8")
    lightgbm = lgb.LGBMClassifier(**LIGHTGBM_CONFIG)
    lightgbm.fit(x_train, y_train)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    assert abs(scale_pos_weight - 27.434310084) < 1e-8
    xgboost = xgb.XGBClassifier(**XGBOOST_CONFIG, scale_pos_weight=scale_pos_weight)
    xgboost.fit(x_train, y_train)
    manifest = write_bundle(
        destination,
        feature_state=feature_state,
        lightgbm_model=lightgbm,
        xgboost_model=xgboost,
        calibrator_path=frozen_calibrator,
        rules_path=frozen_rules,
        lightgbm_config=LIGHTGBM_CONFIG,
        xgboost_config=XGBOOST_CONFIG,
        scale_pos_weight=scale_pos_weight,
        frozen_manifest_path=ROOT / "reports/frozen_system_manifest.md",
        holdout_report_path=ROOT / "reports/final_holdout_evaluation_summary.md",
    )
    sample_matrix = x_train[-100:]
    saved_lightgbm = lgb.Booster(model_file=str(destination / "lightgbm.txt"))
    saved_xgboost = xgb.Booster(model_file=str(destination / "xgboost.json"))
    np.testing.assert_array_equal(
        saved_lightgbm.predict(sample_matrix), lightgbm.predict_proba(sample_matrix)[:, 1]
    )
    np.testing.assert_array_equal(
        saved_xgboost.predict(xgb.DMatrix(sample_matrix)),
        xgboost.predict_proba(sample_matrix)[:, 1],
    )
    print(
        json.dumps(
            {
                "bundle": str(destination),
                "feature_hash": manifest["ordered_feature_names_sha256"],
                "elapsed_seconds": time.perf_counter() - started,
                "native_model_prediction_parity": "exact on 100 TRAIN rows",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
