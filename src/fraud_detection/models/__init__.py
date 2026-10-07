"""Model training, comparison, calibration, and persistence components."""

from fraud_detection.models.catboost_challenge import (
    CatBoostCVResult,
    CatBoostFoldEvaluation,
    PreparedCatBoostData,
    evaluate_catboost_fold,
    prepare_catboost_data,
    summarize_catboost_cv,
)
from fraud_detection.models.comparison import (
    BinaryMetrics,
    SearchResult,
    TrainingOnlyPreprocessor,
    evaluate_probabilities,
    run_temporal_parameter_search,
    save_model_comparison_plots,
    write_model_comparison_report,
)
from fraud_detection.models.tuning import (
    FoldEvaluation,
    LightGBMCVResult,
    PreparedMatrices,
    TemporalCVFold,
    evaluate_lightgbm_fold,
    make_expanding_time_folds,
    prepare_engineered_matrices,
    summarize_lightgbm_cv,
)

__all__ = [
    "BinaryMetrics",
    "CatBoostCVResult",
    "CatBoostFoldEvaluation",
    "FoldEvaluation",
    "LightGBMCVResult",
    "PreparedMatrices",
    "PreparedCatBoostData",
    "SearchResult",
    "TemporalCVFold",
    "TrainingOnlyPreprocessor",
    "evaluate_probabilities",
    "evaluate_lightgbm_fold",
    "evaluate_catboost_fold",
    "make_expanding_time_folds",
    "prepare_engineered_matrices",
    "prepare_catboost_data",
    "run_temporal_parameter_search",
    "save_model_comparison_plots",
    "summarize_lightgbm_cv",
    "summarize_catboost_cv",
    "write_model_comparison_report",
]
