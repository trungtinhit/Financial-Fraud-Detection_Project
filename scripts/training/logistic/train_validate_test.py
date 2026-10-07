# ============================================================
# FRAUD DETECTION - TRAIN / VALIDATION / TEST
# PaySim
#
# Quy trình:
# 1. Đọc PS_remake.csv
# 2. Tạo feature nếu chưa tồn tại
# 3. Chia:
#       Train      = 70%
#       Validation = 15%
#       Test       = 15%
# 4. Huấn luyện Logistic Regression trên Train
# 5. Dùng Validation để tìm Threshold
# 6. KHÔNG dùng Test để chọn threshold
# 7. Đánh giá cuối cùng trên Test
# 8. Lưu model + threshold
# ============================================================


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)

from scripts.config.paths import (
    PROCESSED_DATA_FILE,
    LOGISTIC_MODEL_FILE,
    LOGISTIC_THRESHOLD_FILE,
    LOGISTIC_VALIDATION_THRESHOLD_RESULTS_FILE
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATA_FILE = PROCESSED_DATA_FILE

MODEL_FILE = LOGISTIC_MODEL_FILE

THRESHOLD_FILE = LOGISTIC_THRESHOLD_FILE

THRESHOLD_RESULT_FILE = LOGISTIC_VALIDATION_THRESHOLD_RESULTS_FILE

RANDOM_STATE = 42


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("1. ĐỌC DỮ LIỆU")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("Đọc dữ liệu thành công!")

print(
    f"Số dòng: {len(df):,}"
)

print(
    f"Số cột: {len(df.columns)}"
)


# ============================================================
# 3. TẠO FEATURE NẾU CHƯA CÓ
# ============================================================

print("\n" + "=" * 70)
print("2. CHUẨN BỊ FEATURE")
print("=" * 70)


if "balanceDiffOrig" not in df.columns:

    print("Đang tạo balanceDiffOrig...")

    df["balanceDiffOrig"] = (
        df["oldbalanceOrg"]
        - df["newbalanceOrig"]
    )

else:

    print("balanceDiffOrig đã tồn tại.")


if "balanceDiffDest" not in df.columns:

    print("Đang tạo balanceDiffDest...")

    df["balanceDiffDest"] = (
        df["newbalanceDest"]
        - df["oldbalanceDest"]
    )

else:

    print("balanceDiffDest đã tồn tại.")


# ============================================================
# 4. XÁC ĐỊNH FEATURES
# ============================================================

print("\n" + "=" * 70)
print("3. CHỌN FEATURES")
print("=" * 70)


FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balanceDiffOrig",
    "balanceDiffDest"
]


TARGET = "isFraud"


# Kiểm tra cột

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


X = df[FEATURES]

y = df[TARGET]


print("\nFeatures:")

for feature in FEATURES:

    print(" -", feature)


print("\nTarget:")

print(" -", TARGET)


# ============================================================
# 5. CHIA TRAIN + TEMP
# ============================================================
#
# Train = 70%
# Temp  = 30%
#
# Sau đó Temp tiếp tục chia:
#
# Validation = 15%
# Test       = 15%
#
# ============================================================

print("\n" + "=" * 70)
print("4. CHIA TRAIN / VALIDATION / TEST")
print("=" * 70)


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)


# Temp có 30%.
#
# Chia đôi Temp:
# Validation = 15% toàn bộ
# Test       = 15% toàn bộ

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print(
    f"Train      : {len(X_train):,}"
)

print(
    f"Validation : {len(X_val):,}"
)

print(
    f"Test       : {len(X_test):,}"
)


# ============================================================
# 6. KIỂM TRA TỶ LỆ FRAUD
# ============================================================

print("\n" + "=" * 70)
print("5. KIỂM TRA PHÂN BỐ FRAUD")
print("=" * 70)


print(
    "\nTrain:"
)

print(
    y_train.value_counts()
)

print(
    f"Tỷ lệ fraud: {y_train.mean() * 100:.4f}%"
)


print(
    "\nValidation:"
)

print(
    y_val.value_counts()
)

print(
    f"Tỷ lệ fraud: {y_val.mean() * 100:.4f}%"
)


print(
    "\nTest:"
)

print(
    y_test.value_counts()
)

print(
    f"Tỷ lệ fraud: {y_test.mean() * 100:.4f}%"
)


# ============================================================
# 7. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("6. PREPROCESSING")
print("=" * 70)


categorical_features = [
    "type"
]


numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balanceDiffOrig",
    "balanceDiffDest"
]


preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",

            StandardScaler(),

            numeric_features
        ),

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )
    ]
)


# ============================================================
# 8. MODEL
# ============================================================

print("\n" + "=" * 70)
print("7. TẠO MODEL")
print("=" * 70)


model = LogisticRegression(

    class_weight="balanced",

    max_iter=1000,

    random_state=RANDOM_STATE
)


# ============================================================
# 9. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[

        (
            "preprocessing",

            preprocessor
        ),

        (
            "model",

            model
        )
    ]
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("8. HUẤN LUYỆN MODEL")
print("=" * 70)


print("Đang huấn luyện...")

pipeline.fit(
    X_train,
    y_train
)

print("Huấn luyện hoàn tất!")


# ============================================================
# 11. DỰ ĐOÁN VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("9. DỰ ĐOÁN VALIDATION")
print("=" * 70)


y_val_prob = pipeline.predict_proba(
    X_val
)[:, 1]


print("Đã tính xác suất Validation.")


# ============================================================
# 12. TÍNH PR-AUC VALIDATION
# ============================================================

val_pr_auc = average_precision_score(
    y_val,
    y_val_prob
)


val_roc_auc = roc_auc_score(
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
# 13. TÌM THRESHOLD TRÊN VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("10. TÌM THRESHOLD TRÊN VALIDATION")
print("=" * 70)


# Thử từ 0.50 đến 0.99
# bước 0.01

thresholds = np.arange(
    0.50,
    1.00,
    0.01
)


validation_results = []


for threshold in thresholds:

    y_val_pred = (
        y_val_prob >= threshold
    ).astype(int)


    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )


    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )


    cm = confusion_matrix(
        y_val,
        y_val_pred
    )


    tn, fp, fn, tp = cm.ravel()


    validation_results.append({

        "threshold": threshold,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp
    })


validation_results_df = pd.DataFrame(
    validation_results
)


# ============================================================
# 14. CHỌN THRESHOLD F1 CAO NHẤT
# ============================================================

best_index = validation_results_df[
    "f1"
].idxmax()


best_threshold = float(
    validation_results_df.loc[
        best_index,
        "threshold"
    ]
)


best_precision = float(
    validation_results_df.loc[
        best_index,
        "precision"
    ]
)


best_recall = float(
    validation_results_df.loc[
        best_index,
        "recall"
    ]
)


best_f1 = float(
    validation_results_df.loc[
        best_index,
        "f1"
    ]
)


print("\n" + "=" * 70)
print("11. THRESHOLD ĐƯỢC CHỌN")
print("=" * 70)


print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Precision : {best_precision:.6f}"
)

print(
    f"Recall    : {best_recall:.6f}"
)

print(
    f"F1-score  : {best_f1:.6f}"
)


# ============================================================
# 15. LƯU BẢNG VALIDATION
# ============================================================

validation_results_df.to_csv(
    THRESHOLD_RESULT_FILE,
    index=False
)


print(
    "\nĐã lưu bảng threshold Validation:"
)

print(
    THRESHOLD_RESULT_FILE
)


# ============================================================
# 16. ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST
# ============================================================

print("\n" + "=" * 70)
print("12. ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST")
print("=" * 70)


print(
    "Đang dự đoán Test..."
)


y_test_prob = pipeline.predict_proba(
    X_test
)[:, 1]


# Sử dụng threshold đã được chọn
# từ Validation

y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm_test = confusion_matrix(
    y_test,
    y_test_pred
)


print("\nCONFUSION MATRIX:")

print(
    cm_test
)


tn, fp, fn, tp = cm_test.ravel()


# ============================================================
# 18. CÁC CHỈ SỐ TEST
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)


test_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)


test_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)


test_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)


test_roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)


test_pr_auc = average_precision_score(
    y_test,
    y_test_prob
)


# ============================================================
# 19. IN KẾT QUẢ
# ============================================================

print("\n" + "=" * 70)
print("13. KẾT QUẢ TEST")
print("=" * 70)


print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {test_accuracy:.6f}"
)

print(
    f"Precision : {test_precision:.6f}"
)

print(
    f"Recall    : {test_recall:.6f}"
)

print(
    f"F1-score  : {test_f1:.6f}"
)

print(
    f"ROC-AUC   : {test_roc_auc:.6f}"
)

print(
    f"PR-AUC    : {test_pr_auc:.6f}"
)


# ============================================================
# 20. CONFUSION MATRIX CHI TIẾT
# ============================================================

print("\n" + "=" * 70)
print("14. CHI TIẾT CONFUSION MATRIX")
print("=" * 70)


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
# 21. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("15. CLASSIFICATION REPORT")
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
# 22. LƯU MODEL
# ============================================================

print("\n" + "=" * 70)
print("16. LƯU MODEL")
print("=" * 70)


joblib.dump(
    pipeline,
    MODEL_FILE
)


print(
    f"Đã lưu model:"
)

print(
    MODEL_FILE
)


# ============================================================
# 23. LƯU THRESHOLD
# ============================================================

with open(
    THRESHOLD_FILE,
    "w"
) as f:

    f.write(
        str(best_threshold)
    )


print(
    f"Đã lưu threshold:"
)

print(
    THRESHOLD_FILE
)


# ============================================================
# 24. KẾT THÚC
# ============================================================

print("\n" + "=" * 70)
print("HOÀN TẤT PIPELINE")
print("=" * 70)


print(
    """
PS_remake.csv
      |
      v
Train / Validation / Test
      |
      v
Logistic Regression
      |
      v
Validation
      |
      v
Chọn Threshold
      |
      v
Test cuối cùng
      |
      v
Đánh giá
      |
      v
Lưu Model + Threshold
"""
)