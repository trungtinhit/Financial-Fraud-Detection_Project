# ============================================================
# XGBOOST DEPLOYMENT MODEL
# Financial Fraud Detection
#
# Model triển khai chỉ dùng các thông tin
# có thể biết trước / tại thời điểm giao dịch:
#
# - step
# - type
# - amount
# - oldbalanceOrg
#
# Không dùng:
# - newbalanceOrig
# - oldbalanceDest
# - newbalanceDest
# - balanceDiffOrig
# - balanceDiffDest
# ============================================================

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. IMPORT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from paths import (
    PROCESSED_DATA_FILE,
    XGBOOST_DEPLOYMENT_MODEL_FILE,
    XGBOOST_DEPLOYMENT_THRESHOLD_FILE,
    XGBOOST_DEPLOYMENT_THRESHOLD_RESULTS_FILE
)


# ============================================================
# 2. CẤU HÌNH
# ============================================================

RANDOM_STATE = 42

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg"
]

TARGET = "isFraud"


# ============================================================
# 3. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("1. ĐỌC DỮ LIỆU")
print("=" * 70)

print(f"File dữ liệu: {PROCESSED_DATA_FILE}")

df = pd.read_csv(PROCESSED_DATA_FILE)

print("Đọc dữ liệu thành công!")

print(f"Số dòng: {len(df):,}")
print(f"Số cột : {len(df.columns)}")


# ============================================================
# 4. KIỂM TRA FEATURE
# ============================================================

print("\n" + "=" * 70)
print("2. KIỂM TRA FEATURE")
print("=" * 70)

required_columns = FEATURES + [TARGET]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Thiếu các cột: {missing_columns}"
    )


print("Các feature triển khai:")

for feature in FEATURES:
    print(f" - {feature}")


# ============================================================
# 5. X / y
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()


print("\nPhân bố target:")

print(y.value_counts())


# ============================================================
# 6. TRAIN / VALIDATION / TEST
# ============================================================

print("\n" + "=" * 70)
print("3. CHIA TRAIN / VALIDATION / TEST")
print("=" * 70)

# ------------------------------------------------------------
# Train = 70%
# Temp = 30%
# ------------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# ------------------------------------------------------------
# Validation = 15%
# Test = 15%
# ------------------------------------------------------------

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")

print()
print(f"Fraud Train      : {int(y_train.sum()):,}")
print(f"Fraud Validation : {int(y_val.sum()):,}")
print(f"Fraud Test       : {int(y_test.sum()):,}")


# ============================================================
# 7. XỬ LÝ MẤT CÂN BẰNG
# ============================================================

print("\n" + "=" * 70)
print("4. XỬ LÝ MẤT CÂN BẰNG")
print("=" * 70)

train_normal = int(
    (y_train == 0).sum()
)

train_fraud = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    train_normal / train_fraud
)

print(f"Normal Train     : {train_normal:,}")
print(f"Fraud Train      : {train_fraud:,}")
print(f"scale_pos_weight : {scale_pos_weight:.2f}")


# ============================================================
# 8. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("5. PREPROCESSING")
print("=" * 70)

categorical_features = [
    "type"
]

numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 9. XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("6. TẠO XGBOOST DEPLOYMENT MODEL")
print("=" * 70)

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
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            xgb_model
        )
    ]
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("7. HUẤN LUYỆN MODEL")
print("=" * 70)

print("Đang huấn luyện...")

pipeline.fit(
    X_train,
    y_train
)

print("Huấn luyện hoàn tất!")


# ============================================================
# 12. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("8. VALIDATION")
print("=" * 70)

y_val_prob = pipeline.predict_proba(
    X_val
)[:, 1]

val_roc_auc = roc_auc_score(
    y_val,
    y_val_prob
)

val_pr_auc = average_precision_score(
    y_val,
    y_val_prob
)

print(
    f"Validation ROC-AUC : {val_roc_auc:.6f}"
)

print(
    f"Validation PR-AUC  : {val_pr_auc:.6f}"
)


# ============================================================
# 13. THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 70)
print("9. THRESHOLD TUNING")
print("=" * 70)

thresholds = np.arange(
    0.50,
    1.00,
    0.01
)

results = []


for threshold in thresholds:

    y_pred = (
        y_val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# 14. CHỌN THRESHOLD
# ============================================================

best_index = (
    results_df["f1"]
    .idxmax()
)

best_row = (
    results_df.loc[
        best_index
    ]
)

best_threshold = float(
    best_row["threshold"]
)


print("\n" + "=" * 70)
print("10. THRESHOLD ĐƯỢC CHỌN")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Precision : {best_row['precision']:.6f}"
)

print(
    f"Recall    : {best_row['recall']:.6f}"
)

print(
    f"F1-score  : {best_row['f1']:.6f}"
)


# ============================================================
# 15. LƯU THRESHOLD RESULTS
# ============================================================

XGBOOST_DEPLOYMENT_THRESHOLD_RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    XGBOOST_DEPLOYMENT_THRESHOLD_RESULTS_FILE,
    index=False
)

print()
print("Đã lưu threshold results:")

print(
    XGBOOST_DEPLOYMENT_THRESHOLD_RESULTS_FILE
)


# ============================================================
# 16. TEST
# ============================================================

print("\n" + "=" * 70)
print("11. ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST")
print("=" * 70)

y_test_prob = pipeline.predict_proba(
    X_test
)[:, 1]

y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX:")

print(cm)


# ============================================================
# 18. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)

pr_auc = average_precision_score(
    y_test,
    y_test_prob
)


# ============================================================
# 19. KẾT QUẢ TEST
# ============================================================

print("\n" + "=" * 70)
print("12. KẾT QUẢ TEST - DEPLOYMENT MODEL")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {accuracy:.6f}"
)

print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F1-score  : {f1:.6f}"
)

print(
    f"ROC-AUC   : {roc_auc:.6f}"
)

print(
    f"PR-AUC    : {pr_auc:.6f}"
)


print("\nCHI TIẾT CONFUSION MATRIX:")

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
# 20. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("13. CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Normal",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 21. LƯU MODEL
# ============================================================

print("\n" + "=" * 70)
print("14. LƯU DEPLOYMENT MODEL")
print("=" * 70)

XGBOOST_DEPLOYMENT_MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    pipeline,
    XGBOOST_DEPLOYMENT_MODEL_FILE
)

print(
    "Đã lưu model:"
)

print(
    XGBOOST_DEPLOYMENT_MODEL_FILE
)


# ============================================================
# 22. LƯU THRESHOLD
# ============================================================

with open(
    XGBOOST_DEPLOYMENT_THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"{best_threshold:.2f}"
    )


print()
print(
    "Đã lưu threshold:"
)

print(
    XGBOOST_DEPLOYMENT_THRESHOLD_FILE
)


# ============================================================
# 23. HOÀN TẤT
# ============================================================

print("\n" + "=" * 70)
print("HOÀN TẤT DEPLOYMENT MODEL")
print("=" * 70)