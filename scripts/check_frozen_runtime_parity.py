"""Fresh-process, label-free regression check against terminal locked predictions."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.inference import FrozenFraudInferenceRuntime  # noqa: E402
from fraud_detection.inference.bundle import verify_bundle  # noqa: E402

SCORE_COLUMNS = (
    "lightgbm_score",
    "xgboost_score",
    "ensemble_score",
    "calibrated_probability",
)
# The full TRAIN rebuild produced bit-identical 467-feature matrices for this
# prefix. Native XGBoost scores match the lock to 1e-16. Three LightGBM scores
# differ by at most 7.33e-12 after independent frozen TRAIN-only materialization.
# This narrow absolute bound covers only that observed floating-point variation.
TOLERANCE = 1e-11


def compare_scores(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, float]:
    differences = {}
    for column in SCORE_COLUMNS:
        lhs = actual[column].to_numpy(dtype=np.float64)
        rhs = expected[column].to_numpy(dtype=np.float64)
        differences[column] = float(np.max(np.abs(lhs - rhs)))
        np.testing.assert_allclose(lhs, rhs, rtol=0, atol=TOLERANCE)
    np.testing.assert_array_equal(actual.base_decision, expected.base_decision)
    np.testing.assert_array_equal(actual.TransactionID, expected.TransactionID)
    return differences


def main() -> None:
    bundle_path = ROOT / "models/frozen_v1"
    fixture_path = ROOT / "reports/runtime_golden_input_prefix.parquet"
    rows = pd.read_parquet(fixture_path)
    assert "isFraud" not in rows and "y_true" not in rows
    locked = pd.read_csv(ROOT / "reports/holdout_predictions_frozen.csv", nrows=len(rows))
    assert "isFraud" not in locked
    assert np.array_equal(rows.TransactionID, locked.TransactionID)
    size_bytes = sum(path.stat().st_size for path in bundle_path.iterdir() if path.is_file())
    verify_bundle(bundle_path)
    process = psutil.Process()
    memory_before = process.memory_info().rss
    with patch("pandas.read_csv", side_effect=AssertionError("runtime must not read raw CSV")):
        started = time.perf_counter()
        runtime = FrozenFraudInferenceRuntime.load(bundle_path)
        load_seconds = time.perf_counter() - started
        memory_after_load = process.memory_info().rss
        assert len(runtime.feature_state.feature_names) == 467
        source_copy = rows.copy(deep=True)
        actual = runtime.predict_batch(rows, include_rules=True)
        pd.testing.assert_frame_equal(rows, source_copy)
        memory_after_inference = process.memory_info().rss
        differences = compare_scores(actual, locked)
        np.testing.assert_array_equal(actual.final_decision, locked.optional_rules_decision)
        np.testing.assert_array_equal(
            actual.triggered_rule_ids,
            locked.optional_triggered_rule_ids.fillna("").to_numpy(),
        )
        assert np.all((actual.calibrated_probability >= 0) & (actual.calibrated_probability <= 1))
        np.testing.assert_allclose(
            actual.ensemble_score,
            0.8 * actual.lightgbm_score + 0.2 * actual.xgboost_score,
            rtol=0,
            atol=0,
        )
        second = FrozenFraudInferenceRuntime.load(bundle_path)
        repeated = second.predict_batch(rows)
        compare_scores(repeated, actual)
        np.testing.assert_array_equal(
            repeated.loc[:, list(SCORE_COLUMNS)], actual.loc[:, list(SCORE_COLUMNS)]
        )
        first_three = rows.iloc[:3]
        batch_runtime = FrozenFraudInferenceRuntime.load(bundle_path)
        batch = batch_runtime.predict_batch(first_three)
        single_runtime = FrozenFraudInferenceRuntime.load(bundle_path)
        sequential = pd.DataFrame(
            [single_runtime.predict_one(row) for row in first_three.to_dict(orient="records")]
        )
        compare_scores(batch.reset_index(drop=True), sequential)
        np.testing.assert_array_equal(
            batch.loc[:, list(SCORE_COLUMNS)].to_numpy(),
            sequential.loc[:, list(SCORE_COLUMNS)].to_numpy(),
        )
        batch_runtime.reset_state()
        reset = batch_runtime.predict_batch(first_three)
        compare_scores(batch, reset)
    result = {
        "bundle_path": str(bundle_path),
        "bundle_bytes": size_bytes,
        "feature_count": 467,
        "golden_prefix_rows": len(rows),
        "golden_max_absolute_differences": differences,
        "score_tolerance_absolute": TOLERANCE,
        "base_decisions_exact": True,
        "optional_rules_decisions_and_triggers_exact": True,
        "fresh_process_no_raw_train_csv": True,
        "independent_reload_parity": True,
        "batch_vs_sequential_rows": len(first_three),
        "batch_vs_sequential_parity": True,
        "reset_parity": True,
        "load_seconds": load_seconds,
        "feature_seconds": runtime.last_timings_["feature_seconds"],
        "model_seconds": runtime.last_timings_["model_seconds"],
        "calibration_decision_seconds": runtime.last_timings_["calibration_decision_seconds"],
        "total_batch_seconds": runtime.last_timings_["total_seconds"],
        "throughput_rows_per_second": len(rows) / runtime.last_timings_["total_seconds"],
        "rss_before_bytes": memory_before,
        "rss_after_load_bytes": memory_after_load,
        "rss_after_inference_bytes": memory_after_inference,
    }
    destination = ROOT / "reports/frozen_runtime_parity_benchmark.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
