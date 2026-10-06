# ============================================================
# XGBOOST DEPLOYMENT MODEL V3
#
# Features:
# - type
# - amount
# - oldbalanceOrg
# - transactions_per_hour
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
    V3_DATA_FILE,
    XGBOOST_DEPLOYMENT_V3_MODEL_FILE,
    XGBOOST_DEPLOYMENT_V3_THRESHOLD_FILE,
    XGBOOST_DEPLOYMENT_V3_THRESHOLD_RESULTS_FILE
)


# ============================================================
# 1. CONFIG
# ============================================================

RANDOM_STATE = 42


FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "transactions_per_hour"
]


CATEGORICAL_FEATURES = [
    "type"
]


NUMERICAL_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "transactions_per_hour"
]


TARGET = "isFraud"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("XGBOOST DEPLOYMENT MODEL V3")
print("=" * 70)

print("\nDataset:")
print(V3_DATA_FILE)


df = pd.read_csv(
    V3_DATA_FILE,
    usecols=FEATURES + [TARGET]
)


print("\nDataset shape:")
print(df.shape)


print("\nFeatures:")

for feature in FEATURES:
    print(f"- {feature}")


# ============================================================
# 3. X / Y
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].copy()


print("\nTarget distribution:")

print(
    y.value_counts()
)


# ============================================================
# 4. TRAIN / TEMP
#
# 70% train
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
# 5. VALIDATION / TEST
#
# 15% validation
# 15% test
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
# 6. CLASS IMBALANCE
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
# 7. PREPROCESSOR
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
# 8. XGBOOST
#
# Giữ hyperparameter giống V2
# để so sánh V2 và V3 công bằng.
# ============================================================

classifier = XGBClassifier(

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
# 9. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )
    ]
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\nĐang huấn luyện Deployment V3...")


pipeline.fit(
    X_train,
    y_train
)


print(
    "Huấn luyện hoàn tất."
)


# ============================================================
# 11. VALIDATION PROBABILITY
# ============================================================

val_prob = (
    pipeline
    .predict_proba(
        X_val
    )[:, 1]
)


# ============================================================
# 12. VALIDATION ROC / PR
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
print("VALIDATION")
print("=" * 70)


print(
    f"ROC-AUC: {val_roc_auc:.6f}"
)

print(
    f"PR-AUC:  {val_pr_auc:.6f}"
)


# ============================================================
# 13. THRESHOLD TUNING
# ============================================================

thresholds = np.arange(
    0.50,
    1.00,
    0.01
)


threshold_results = []


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
# 14. BEST THRESHOLD
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
# 15. SAVE THRESHOLD RESULTS
# ============================================================

XGBOOST_DEPLOYMENT_V3_THRESHOLD_RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


threshold_df.to_csv(
    XGBOOST_DEPLOYMENT_V3_THRESHOLD_RESULTS_FILE,
    index=False
)


# ============================================================
# 16. FINAL TEST
# ============================================================

test_prob = (
    pipeline
    .predict_proba(
        X_test
    )[:, 1]
)


test_pred = (
    test_prob >= best_threshold
).astype(int)


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
# 17. PRINT FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULT - DEPLOYMENT V3")
print("=" * 70)


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
# 18. SAVE MODEL
# ============================================================

XGBOOST_DEPLOYMENT_V3_MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    pipeline,
    XGBOOST_DEPLOYMENT_V3_MODEL_FILE
)


print("\nModel saved:")

print(
    XGBOOST_DEPLOYMENT_V3_MODEL_FILE
)


# ============================================================
# 19. SAVE THRESHOLD
# ============================================================

with open(
    XGBOOST_DEPLOYMENT_V3_THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        str(
            best_threshold
        )
    )


print("\nThreshold saved:")

print(
    XGBOOST_DEPLOYMENT_V3_THRESHOLD_FILE
)


print("\n" + "=" * 70)
print("HOÀN THÀNH DEPLOYMENT V3")
print("=" * 70)