from pathlib import Path

# ============================================================
# ROOT PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATA
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv"

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "PS_remake.csv"


# ============================================================
# MODELS
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"

LOGISTIC_MODEL_DIR = MODELS_DIR / "logistic"
RANDOM_FOREST_MODEL_DIR = MODELS_DIR / "random_forest"
XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost"


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"

LOGISTIC_RESULTS_DIR = RESULTS_DIR / "logistic"
RANDOM_FOREST_RESULTS_DIR = RESULTS_DIR / "random_forest"
XGBOOST_RESULTS_DIR = RESULTS_DIR / "xgboost"


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

LOGISTIC_MODEL_FILE = (
    LOGISTIC_MODEL_DIR / "fraud_detection_model_v2.pkl"
)

LOGISTIC_LEGACY_MODEL_FILE = (
    LOGISTIC_MODEL_DIR / "fraud_detection_model.pkl"
)

LOGISTIC_THRESHOLD_FILE = (
    LOGISTIC_MODEL_DIR / "fraud_threshold.txt"
)

LOGISTIC_THRESHOLD_RESULTS_FILE = (
    LOGISTIC_RESULTS_DIR / "threshold_results.csv"
)

LOGISTIC_VALIDATION_THRESHOLD_RESULTS_FILE = (
    LOGISTIC_RESULTS_DIR / "validation_threshold_results.csv"
)


# ============================================================
# RANDOM FOREST
# ============================================================

RANDOM_FOREST_MODEL_FILE = (
    RANDOM_FOREST_MODEL_DIR / "random_forest_fraud_model.pkl"
)

RANDOM_FOREST_THRESHOLD_FILE = (
    RANDOM_FOREST_MODEL_DIR / "random_forest_threshold.txt"
)

RANDOM_FOREST_THRESHOLD_RESULTS_FILE = (
    RANDOM_FOREST_RESULTS_DIR / "random_forest_threshold_results.csv"
)


# ============================================================
# XGBOOST
# ============================================================

XGBOOST_MODEL_FILE = (
    XGBOOST_MODEL_DIR / "xgboost_fraud_model.pkl"
)

XGBOOST_THRESHOLD_FILE = (
    XGBOOST_MODEL_DIR / "xgboost_threshold.txt"
)

XGBOOST_THRESHOLD_RESULTS_FILE = (
    XGBOOST_RESULTS_DIR / "xgboost_threshold_results.csv"
)

XGBOOST_FEATURE_IMPORTANCE_FILE = (
    XGBOOST_RESULTS_DIR / "xgboost_feature_importance.csv"
)

XGBOOST_FEATURE_IMPORTANCE_GROUPED_FILE = (
    XGBOOST_RESULTS_DIR / "xgboost_feature_importance_grouped.csv"
)

# ============================================================
# XGBOOST DEPLOYMENT
# ============================================================

XGBOOST_DEPLOYMENT_MODEL_FILE = (
    XGBOOST_MODEL_DIR / "xgboost_deployment_model.pkl"
)

XGBOOST_DEPLOYMENT_THRESHOLD_FILE = (
    XGBOOST_MODEL_DIR / "xgboost_deployment_threshold.txt"
)

XGBOOST_DEPLOYMENT_THRESHOLD_RESULTS_FILE = (
    XGBOOST_RESULTS_DIR / "xgboost_deployment_threshold_results.csv"
)


# ============================================================
# PLOTS
# ============================================================

PLOTS_DIR = PROJECT_ROOT / "plots"

MODEL_COMPARISON_METRICS_PLOT = (
    PLOTS_DIR / "model_comparison_metrics.png"
)

MODEL_COMPARISON_FP_FN_PLOT = (
    PLOTS_DIR / "model_comparison_fp_fn.png"
)

XGBOOST_FEATURE_IMPORTANCE_PLOT = (
    PLOTS_DIR / "xgboost_feature_importance.png"
)

XGBOOST_FEATURE_IMPORTANCE_GROUPED_PLOT = (
    PLOTS_DIR / "xgboost_feature_importance_grouped.png"
)


# ============================================================
# REPORTS
# ============================================================

REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_COMPARISON_RESULTS_FILE = (
    REPORTS_DIR / "model_comparison_results.csv"
)


# ============================================================
# KIỂM TRA
# ============================================================

if __name__ == "__main__":

    print("PROJECT_ROOT:")
    print(PROJECT_ROOT)

    print("\nPROCESSED_DATA_FILE:")
    print(PROCESSED_DATA_FILE)

    print("\nXGBOOST_MODEL_FILE:")
    print(XGBOOST_MODEL_FILE)

    print("\nKiểm tra file dữ liệu:")

    if PROCESSED_DATA_FILE.exists():
        print("OK - PS_remake.csv tồn tại")
    else:
        print("ERROR - Không tìm thấy PS_remake.csv")