import inspect

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models import (
    aggregate_oof_predictions,
    ensemble,
    evaluate_probability_blends,
    prediction_correlations,
    weighted_score_blend,
)


def model_oof(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [101, 102, 103, 104, 105, 106, 107, 108],
            "fold": [1, 1, 1, 1, 2, 2, 2, 2],
            "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
            "score": scores,
        }
    )


def combined_oof() -> pd.DataFrame:
    return aggregate_oof_predictions(
        model_oof([0.1, 0.8, 0.2, 0.7, 0.15, 0.75, 0.25, 0.85]),
        model_oof([0.2, 0.7, 0.1, 0.8, 0.25, 0.65, 0.15, 0.90]),
    )


def test_weighted_blend_formula_and_weights() -> None:
    blended = weighted_score_blend(
        [0.2, 0.8], [0.6, 0.4], lightgbm_weight=0.75, xgboost_weight=0.25
    )

    np.testing.assert_allclose(blended, [0.3, 0.7])


@pytest.mark.parametrize(
    ("lightgbm_weight", "xgboost_weight", "message"),
    [(0.8, 0.3, "sum"), (-0.1, 1.1, "non-negative"), (np.nan, np.nan, "finite")],
)
def test_invalid_blend_weights_raise(
    lightgbm_weight: float, xgboost_weight: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        weighted_score_blend(
            [0.2],
            [0.3],
            lightgbm_weight=lightgbm_weight,
            xgboost_weight=xgboost_weight,
        )


def test_oof_aggregation_aligns_rows_and_preserves_folds() -> None:
    xgboost = model_oof([0.2, 0.7, 0.1, 0.8, 0.25, 0.65, 0.15, 0.90]).sample(frac=1, random_state=7)

    combined = aggregate_oof_predictions(
        model_oof([0.1, 0.8, 0.2, 0.7, 0.15, 0.75, 0.25, 0.85]), xgboost
    )

    assert combined["TransactionID"].tolist() == [101, 102, 103, 104, 105, 106, 107, 108]
    assert combined["fold"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_oof_aggregation_rejects_missing_or_inconsistent_rows() -> None:
    lightgbm = model_oof([0.1, 0.8, 0.2, 0.7, 0.15, 0.75, 0.25, 0.85])
    missing = model_oof([0.2, 0.7, 0.1, 0.8, 0.25, 0.65, 0.15, 0.90]).iloc[:-1]
    with pytest.raises(ValueError, match="not aligned exactly"):
        aggregate_oof_predictions(lightgbm, missing)

    inconsistent = model_oof([0.2, 0.7, 0.1, 0.8, 0.25, 0.65, 0.15, 0.90])
    inconsistent.loc[0, "fold"] = 2
    with pytest.raises(ValueError, match="not aligned exactly"):
        aggregate_oof_predictions(lightgbm, inconsistent)


def test_missing_prediction_is_rejected() -> None:
    frame = combined_oof()
    frame.loc[0, "xgboost_score"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        evaluate_probability_blends(frame, [1.0, 0.5])


def test_blend_evaluation_is_deterministic_and_uses_supplied_oof_only() -> None:
    frame = combined_oof()
    first = evaluate_probability_blends(frame, [1.0, 0.5])
    second = evaluate_probability_blends(frame.copy(), [1.0, 0.5])

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(prediction_correlations(frame), prediction_correlations(frame))


def test_target_is_not_an_input_to_blend_scores() -> None:
    frame = combined_oof()
    scores_before = weighted_score_blend(
        frame["lightgbm_score"],
        frame["xgboost_score"],
        lightgbm_weight=0.8,
        xgboost_weight=0.2,
    )
    frame["y_true"] = 1 - frame["y_true"]
    scores_after = weighted_score_blend(
        frame["lightgbm_score"],
        frame["xgboost_score"],
        lightgbm_weight=0.8,
        xgboost_weight=0.2,
    )

    np.testing.assert_array_equal(scores_before, scores_after)


def test_selection_utilities_have_no_external_validation_dependency() -> None:
    source = inspect.getsource(ensemble)

    assert "x_validation" not in source
    assert "holdout" not in source.lower()
