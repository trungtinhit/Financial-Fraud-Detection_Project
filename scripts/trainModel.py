# ============================================================
# HỆ THỐNG PHÁT HIỆN GIAO DỊCH GIAN LẬN - PaySim
# ============================================================
#
# Dữ liệu đầu vào:
# PS_remake.csv
#
# Đã loại:
# - nameOrig
# - nameDest
#
# Đã tạo:
# - balanceDiffOrig
# - balanceDiffDest
#
# Quy trình:
# 1. Đọc dữ liệu
# 2. Kiểm tra dữ liệu
# 3. Chọn Feature và Target
# 4. Chia Train/Test
# 5. One-Hot Encoding
# 6. Scaling
# 7. Xử lý mất cân bằng bằng class_weight
# 8. Huấn luyện Logistic Regression
# 9. Đánh giá
# 10. Lưu model
# ============================================================


import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    roc_auc_score
)

from paths import PROCESSED_DATA_FILE, LOGISTIC_LEGACY_MODEL_FILE


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATA_FILE = PROCESSED_DATA_FILE

MODEL_FILE = LOGISTIC_LEGACY_MODEL_FILE

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("1. ĐỌC DỮ LIỆU")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("Đọc dữ liệu thành công!")

print(f"Số dòng: {len(df):,}")
print(f"Số cột ban đầu: {len(df.columns)}")

print("\nCác cột ban đầu:")
print(df.columns.tolist())


# ============================================================
# TẠO 2 FEATURE NẾU FILE CHƯA CÓ
# ============================================================

print("\n" + "=" * 70)
print("TẠO FEATURE")
print("=" * 70)


# Feature chênh lệch số dư người gửi
if "balanceDiffOrig" not in df.columns:

    print("Đang tạo balanceDiffOrig...")

    df["balanceDiffOrig"] = (
        df["oldbalanceOrg"] -
        df["newbalanceOrig"]
    )

else:

    print("balanceDiffOrig đã tồn tại.")


# Feature chênh lệch số dư người nhận
if "balanceDiffDest" not in df.columns:

    print("Đang tạo balanceDiffDest...")

    df["balanceDiffDest"] = (
        df["newbalanceDest"] -
        df["oldbalanceDest"]
    )

else:

    print("balanceDiffDest đã tồn tại.")


print("\nCác cột sau khi chuẩn bị feature:")
print(df.columns.tolist())


# ============================================================
# 3. KIỂM TRA CẤU TRÚC DỮ LIỆU
# ============================================================

print("\n" + "=" * 70)
print("2. KIỂM TRA DỮ LIỆU")
print("=" * 70)

print("\nKiểu dữ liệu:")
print(df.dtypes)

print("\nSố lượng giá trị thiếu:")
print(df.isnull().sum())

print("\nSố lượng dòng trùng:")
print(df.duplicated().sum())


# ============================================================
# 4. KIỂM TRA TARGET isFraud
# ============================================================

print("\n" + "=" * 70)
print("3. PHÂN BỐ isFraud")
print("=" * 70)

fraud_count = df["isFraud"].value_counts()

print(fraud_count)

normal_count = int((df["isFraud"] == 0).sum())
fraud_count_value = int((df["isFraud"] == 1).sum())

total = len(df)

print(f"\nGiao dịch bình thường : {normal_count:,}")
print(f"Giao dịch gian lận    : {fraud_count_value:,}")

print(
    f"Tỷ lệ gian lận        : "
    f"{fraud_count_value / total * 100:.4f}%"
)


# ============================================================
# 5. KIỂM TRA TYPE
# ============================================================

print("\n" + "=" * 70)
print("4. PHÂN BỐ TYPE")
print("=" * 70)

print(df["type"].value_counts())

print("\nQuan hệ giữa TYPE và FRAUD:")

type_fraud = pd.crosstab(
    df["type"],
    df["isFraud"]
)

print(type_fraud)


# ============================================================
# 6. KHÔNG SỬ DỤNG isFlaggedFraud
# ============================================================
#
# isFlaggedFraud là cờ cảnh báo có sẵn trong PaySim.
#
# Ta không sử dụng nó trong model chính để tránh việc
# mô hình dựa vào một kết quả cảnh báo đã có sẵn.
#
# Tuy nhiên vẫn giữ cột trong file gốc.
# ============================================================


# ============================================================
# 7. XÁC ĐỊNH FEATURES VÀ TARGET
# ============================================================

print("\n" + "=" * 70)
print("5. CHỌN FEATURES")
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


# Kiểm tra xem tất cả cột cần thiết có tồn tại không

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Thiếu các cột: {missing_columns}"
    )


X = df[FEATURES]

y = df[TARGET]


print("\nFeatures được sử dụng:")

for feature in FEATURES:
    print(" -", feature)

print("\nTarget:")
print(" -", TARGET)


# ============================================================
# 8. CHIA TRAIN / TEST
# ============================================================

print("\n" + "=" * 70)
print("6. CHIA TRAIN / TEST")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print(f"Train: {len(X_train):,} dòng")
print(f"Test : {len(X_test):,} dòng")


print("\nPhân bố TRAIN:")

print(
    y_train.value_counts()
)

print("\nTỷ lệ fraud TRAIN:")

print(
    y_train.mean() * 100,
    "%"
)


print("\nPhân bố TEST:")

print(
    y_test.value_counts()
)

print("\nTỷ lệ fraud TEST:")

print(
    y_test.mean() * 100,
    "%"
)


# ============================================================
# 9. XÁC ĐỊNH CỘT CATEGORICAL VÀ NUMERIC
# ============================================================

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


# ============================================================
# 10. PREPROCESSING
# ============================================================
#
# type:
#     One-Hot Encoding
#
# numeric:
#     StandardScaler
#
# ColumnTransformer đảm bảo preprocessing được học
# CHỈ trên TRAIN.
# ============================================================

print("\n" + "=" * 70)
print("7. XÂY DỰNG PREPROCESSING")
print("=" * 70)


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
# 11. XÂY DỰNG MODEL
# ============================================================
#
# class_weight="balanced"
#
# Đây là cách xử lý mất cân bằng đầu tiên.
#
# Không dùng SMOTE ở đây vì dataset có hơn 6 triệu dòng.
# ============================================================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE
)

# ============================================================
# 12. TẠO PIPELINE
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
# 13. HUẤN LUYỆN MODEL
# ============================================================

print("\n" + "=" * 70)
print("8. HUẤN LUYỆN MODEL")
print("=" * 70)

print("Đang huấn luyện...")
print("Dataset khá lớn nên có thể mất thời gian.")

pipeline.fit(
    X_train,
    y_train
)

print("\nHuấn luyện hoàn tất!")


# ============================================================
# 14. DỰ ĐOÁN
# ============================================================

print("\n" + "=" * 70)
print("9. DỰ ĐOÁN TRÊN TEST")
print("=" * 70)

y_pred = pipeline.predict(X_test)

# Xác suất dự đoán lớp 1
y_prob = pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("10. CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 16. TÍNH CÁC CHỈ SỐ
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)


print("\n" + "=" * 70)
print("11. KẾT QUẢ")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall   : {recall:.6f}"
)

print(
    f"F1-score : {f1:.6f}"
)

print(
    f"ROC-AUC  : {roc_auc:.6f}"
)

print(
    f"PR-AUC   : {pr_auc:.6f}"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("12. CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. LƯU MODEL
# ============================================================

print("\n" + "=" * 70)
print("13. LƯU MODEL")
print("=" * 70)

joblib.dump(
    pipeline,
    MODEL_FILE
)

print(
    f"Đã lưu model tại: {MODEL_FILE}"
)


# ============================================================
# 19. HOÀN TẤT
# ============================================================

print("\n" + "=" * 70)
print("HOÀN TẤT")
print("=" * 70)

print(
    """
Pipeline đã hoàn thành:

CSV
 ↓
Train/Test Split
 ↓
One-Hot Encoding
 ↓
StandardScaler
 ↓
Class Weight
 ↓
Logistic Regression
 ↓
Evaluation
 ↓
Saved Model
"""
)