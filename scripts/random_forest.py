# ============================================================
# RANDOM FOREST - FRAUD DETECTION - PaySim
#
# Quy trình:
# 1. Đọc PS_remake.csv
# 2. Tạo features
# 3. Chia Train 70% / Validation 15% / Test 15%
# 4. Random Forest
# 5. Validation để tìm threshold
# 6. Test để đánh giá cuối cùng
# 7. Lưu model
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

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

from paths import (
    PROCESSED_DATA_FILE,
    RANDOM_FOREST_MODEL_FILE,
    RANDOM_FOREST_THRESHOLD_FILE,
    RANDOM_FOREST_THRESHOLD_RESULTS_FILE
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATA_FILE = PROCESSED_DATA_FILE

MODEL_FILE = RANDOM_FOREST_MODEL_FILE

THRESHOLD_FILE = RANDOM_FOREST_THRESHOLD_FILE

THRESHOLD_RESULT_FILE = RANDOM_FOREST_THRESHOLD_RESULTS_FILE

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
# 3. TẠO FEATURE
# ============================================================

print("\n" + "=" * 70)
print("2. CHUẨN BỊ FEATURE")
print("=" * 70)


if "balanceDiffOrig" not in df.columns:

    df["balanceDiffOrig"] = (
        df["oldbalanceOrg"]
        - df["newbalanceOrig"]
    )

    print("Đã tạo balanceDiffOrig.")

else:

    print("balanceDiffOrig đã tồn tại.")


if "balanceDiffDest" not in df.columns:

    df["balanceDiffDest"] = (
        df["newbalanceDest"]
        - df["oldbalanceDest"]
    )

    print("Đã tạo balanceDiffDest.")

else:

    print("balanceDiffDest đã tồn tại.")


# ============================================================
# 4. FEATURES
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


# ============================================================
# 5. CHIA TRAIN / VALIDATION / TEST
# ============================================================

print("\n" + "=" * 70)
print("4. CHIA TRAIN / VALIDATION / TEST")
print("=" * 70)


# Train = 70%
# Temp = 30%

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=y
)


# Temp:
# Validation = 15%
# Test = 15%

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
# 6. PREPROCESSING
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
# 7. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("6. TẠO RANDOM FOREST")
print("=" * 70)


rf_model = RandomForestClassifier(

    n_estimators=100,

    max_depth=15,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


# ============================================================
# 8. PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessing",

            preprocessor
        ),

        (
            "model",

            rf_model
        )
    ]
)


# ============================================================
# 9. HUẤN LUYỆN
# ============================================================

print("\n" + "=" * 70)
print("7. HUẤN LUYỆN RANDOM FOREST")
print("=" * 70)

print(
    "Đang huấn luyện..."
)

print(
    "Có thể mất nhiều thời gian do dataset lớn."
)


pipeline.fit(

    X_train,

    y_train
)


print(
    "Huấn luyện hoàn tất!"
)


# ============================================================
# 10. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("8. DỰ ĐOÁN VALIDATION")
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
# 11. THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 70)
print("9. TÌM THRESHOLD TRÊN VALIDATION")
print("=" * 70)


thresholds = np.arange(

    0.50,

    1.00,

    0.01
)


results = []


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


    tn, fp, fn, tp = confusion_matrix(

        y_val,

        y_val_pred

    ).ravel()


    results.append({

        "threshold": threshold,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp

    })


results_df = pd.DataFrame(results)


# ============================================================
# 12. CHỌN THRESHOLD F1 CAO NHẤT
# ============================================================

best_index = results_df["f1"].idxmax()


best_threshold = float(

    results_df.loc[
        best_index,
        "threshold"
    ]

)


best_precision = float(

    results_df.loc[
        best_index,
        "precision"
    ]

)


best_recall = float(

    results_df.loc[
        best_index,
        "recall"
    ]

)


best_f1 = float(

    results_df.loc[
        best_index,
        "f1"
    ]

)


print("\n" + "=" * 70)
print("10. THRESHOLD TỐT NHẤT TRÊN VALIDATION")
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
# 13. LƯU KẾT QUẢ THRESHOLD
# ============================================================

results_df.to_csv(

    THRESHOLD_RESULT_FILE,

    index=False
)


print(
    "\nĐã lưu:"
)

print(
    THRESHOLD_RESULT_FILE
)


# ============================================================
# 14. DỰ ĐOÁN TEST
# ============================================================

print("\n" + "=" * 70)
print("11. ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST")
print("=" * 70)


print(
    "Đang dự đoán Test..."
)


y_test_prob = pipeline.predict_proba(

    X_test

)[:, 1]


# QUAN TRỌNG:
# dùng threshold lấy từ Validation

y_test_pred = (

    y_test_prob >= best_threshold

).astype(int)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_test_pred
)


print("\nCONFUSION MATRIX:")

print(cm)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# 16. METRICS
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
# 17. IN KẾT QUẢ
# ============================================================

print("\n" + "=" * 70)
print("12. KẾT QUẢ TEST - RANDOM FOREST")
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
# 18. CONFUSION MATRIX CHI TIẾT
# ============================================================

print("\n" + "=" * 70)
print("13. CHI TIẾT CONFUSION MATRIX")
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
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("14. CLASSIFICATION REPORT")
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
# 20. LƯU MODEL
# ============================================================

print("\n" + "=" * 70)
print("15. LƯU MODEL")
print("=" * 70)


joblib.dump(

    pipeline,

    MODEL_FILE

)


print(
    f"Đã lưu model: {MODEL_FILE}"
)


# ============================================================
# 21. LƯU THRESHOLD
# ============================================================

with open(

    THRESHOLD_FILE,

    "w"

) as f:

    f.write(

        str(best_threshold)

    )


print(
    f"Đã lưu threshold: {THRESHOLD_FILE}"
)


# ============================================================
# 22. HOÀN TẤT
# ============================================================

print("\n" + "=" * 70)
print("HOÀN TẤT RANDOM FOREST")
print("=" * 70)