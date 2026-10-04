# ============================================================
# XGBOOST DEPLOYMENT MODEL V2
#
# Features:
# - type
# - amount
# - oldbalanceOrg
#
# Không sử dụng:
# - step
# - balance sau giao dịch
# - balance người nhận
# ============================================================

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from paths import (
    PROCESSED_DATA_FILE,
    XGBOOST_DEPLOYMENT_V2_MODEL_FILE,
    XGBOOST_DEPLOYMENT_V2_THRESHOLD_FILE,
    XGBOOST_DEPLOYMENT_V2_THRESHOLD_RESULTS_FILE
)


# ============================================================
# 1. CONFIG
# ============================================================

RANDOM_STATE = 42


FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg"
]


CATEGORICAL_FEATURES = [
    "type"
]


NUMERICAL_FEATURES = [
    "amount",
    "oldbalanceOrg"
]


TARGET = "isFraud"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("XGBOOST DEPLOYMENT MODEL V2 - 3 FEATURES")
print("=" * 70)

print("\nĐang đọc dữ liệu:")
print(PROCESSED_DATA_FILE)


df = pd.read_csv(
    PROCESSED_DATA_FILE,
    usecols=FEATURES + [TARGET]
)


print("\nKích thước dữ liệu:")
print(df.shape)


print("\nCác cột:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    df.isnull().sum()
)


# ============================================================
# 4. FEATURES / TARGET
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].copy()


print("\nFeature sử dụng:")

for feature in FEATURES:
    print(f"- {feature}")


print("\nPhân bố target:")

print(
    y.value_counts()
)


# ============================================================
# 5. TRAIN / TEMP SPLIT
#
# 70% train
# 30% temp
# ============================================================

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


# ============================================================
# 6. VALIDATION / TEST
#
# temp 30%
# -> validation 15%
# -> test 15%
# ============================================================

X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
)


print("\nDataset split:")

print(
    f"Train:      {len(X_train):,}"
)

print(
    f"Validation: {len(X_val):,}"
)

print(
    f"Test:       {len(X_test):,}"
)


# ============================================================
# 7. SCALE POS WEIGHT
# ============================================================

train_counts = (
    y_train.value_counts()
)


normal_count = train_counts.get(
    0,
    0
)

fraud_count = train_counts.get(
    1,
    0
)


scale_pos_weight = (
    normal_count /
    fraud_count
)


print("\nTrain class distribution:")

print(
    f"Normal: {normal_count:,}"
)

print(
    f"Fraud:  {fraud_count:,}"
)

print(
    f"scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# 8. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES
        ),

        (
            "numeric",
            "passthrough",
            NUMERICAL_FEATURES
        )
    ]
)


# ============================================================
# 9. XGBOOST
#
# Giữ cùng hyperparameters với model trước
# để phép so sánh công bằng hơn.
# ============================================================

xgb_model = XGBClassifier(

    n_estimators=300,

    max_depth=8,

    learning_rate=0.10,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=2,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


# ============================================================
# 10. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            xgb_model
        )
    ]
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\nĐang huấn luyện model...")

pipeline.fit(
    X_train,
    y_train
)

print(
    "Huấn luyện hoàn tất."
)


# ============================================================
# 12. VALIDATION PROBABILITY
# ============================================================

val_prob = (
    pipeline.predict_proba(
        X_val
    )[:, 1]
)


# ============================================================
# 13. VALIDATION ROC-AUC / PR-AUC
# ============================================================

val_roc_auc = roc_auc_score(
    y_val,
    val_prob
)

val_pr_auc = average_precision_score(
    y_val,
    val_prob
)


print("\n" + "=" * 70)
print("VALIDATION METRICS")
print("=" * 70)

print(
    f"ROC-AUC: {val_roc_auc:.6f}"
)

print(
    f"PR-AUC:  {val_pr_auc:.6f}"
)


# ============================================================
# 14. THRESHOLD TUNING
#
# Thử từ 0.50 đến 0.99
# Chọn threshold có F1 cao nhất
# ============================================================

thresholds = np.arange(
    0.50,
    1.00,
    0.01
)


threshold_results = []


print("\nĐang tìm threshold tốt nhất...")


for threshold in thresholds:

    val_pred = (
        val_prob >= threshold
    ).astype(int)


    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            y_val,
            val_pred
        ).ravel()
    )


    threshold_results.append(
        {
            "threshold":
                threshold,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "tn":
                tn,

            "fp":
                fp,

            "fn":
                fn,

            "tp":
                tp
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# 15. SELECT BEST THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df[
        "f1"
    ].idxmax()
]


best_threshold = float(
    best_row[
        "threshold"
    ]
)


print("\n" + "=" * 70)
print("BEST VALIDATION THRESHOLD")
print("=" * 70)


print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Precision: "
    f"{best_row['precision']:.6f}"
)

print(
    f"Recall: "
    f"{best_row['recall']:.6f}"
)

print(
    f"F1: "
    f"{best_row['f1']:.6f}"
)

print(
    f"FP: "
    f"{int(best_row['fp'])}"
)

print(
    f"FN: "
    f"{int(best_row['fn'])}"
)

print(
    f"TP: "
    f"{int(best_row['tp'])}"
)


# ============================================================
# 16. SAVE THRESHOLD RESULTS
# ============================================================

XGBOOST_DEPLOYMENT_V2_THRESHOLD_RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


threshold_df.to_csv(
    XGBOOST_DEPLOYMENT_V2_THRESHOLD_RESULTS_FILE,
    index=False
)


print(
    "\nThreshold results đã lưu:"
)

print(
    XGBOOST_DEPLOYMENT_V2_THRESHOLD_RESULTS_FILE
)


# ============================================================
# 17. TEST SET
# ============================================================

test_prob = (
    pipeline.predict_proba(
        X_test
    )[:, 1]
)


test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# 18. FINAL TEST METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_prob
)

pr_auc = average_precision_score(
    y_test,
    test_prob
)


tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        test_pred
    ).ravel()
)


# ============================================================
# 19. PRINT FINAL TEST RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULT - DEPLOYMENT V2")
print("=" * 70)


print(
    f"Features: "
    f"{FEATURES}"
)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)


print("\nMetrics:")

print(
    f"Accuracy:  {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall:    {recall:.6f}"
)

print(
    f"F1-score:  {f1:.6f}"
)

print(
    f"ROC-AUC:   {roc_auc:.6f}"
)

print(
    f"PR-AUC:    {pr_auc:.6f}"
)


print("\nConfusion Matrix:")

print(
    f"TN = {tn:,}"
)

print(
    f"FP = {fp:,}"
)

print(
    f"FN = {fn:,}"
)

print(
    f"TP = {tp:,}"
)


# ============================================================
# 20. SAVE MODEL
# ============================================================

XGBOOST_DEPLOYMENT_V2_MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    pipeline,
    XGBOOST_DEPLOYMENT_V2_MODEL_FILE
)


print(
    "\nModel đã lưu:"
)

print(
    XGBOOST_DEPLOYMENT_V2_MODEL_FILE
)


# ============================================================
# 21. SAVE THRESHOLD
# ============================================================

with open(
    XGBOOST_DEPLOYMENT_V2_THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        str(
            best_threshold
        )
    )


print(
    "\nThreshold đã lưu:"
)

print(
    XGBOOST_DEPLOYMENT_V2_THRESHOLD_FILE
)


# ============================================================
# 22. FINISH
# ============================================================

print("\n" + "=" * 70)
print("HOÀN THÀNH")
print("=" * 70)