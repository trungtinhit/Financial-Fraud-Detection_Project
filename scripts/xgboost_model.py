# ============================================================
# XGBOOST - FRAUD DETECTION - PaySim
#
# Quy trình:
# 1. Đọc PS_remake.csv
# 2. Tạo feature
# 3. Train / Validation / Test
# 4. XGBoost
# 5. Validation -> tìm threshold
# 6. Test -> đánh giá cuối cùng
# 7. Lưu model
# ============================================================

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
# IMPORT ĐƯỜNG DẪN PROJECT
# ============================================================

from paths import (
    PROCESSED_DATA_FILE,
    XGBOOST_MODEL_DIR,
    XGBOOST_RESULTS_DIR,
    XGBOOST_MODEL_FILE,
    XGBOOST_THRESHOLD_FILE,
    XGBOOST_THRESHOLD_RESULTS_FILE
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

RANDOM_STATE = 42


# ============================================================
# TẠO THƯ MỤC NẾU CHƯA TỒN TẠI
# ============================================================

XGBOOST_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

XGBOOST_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ĐƯỜNG DẪN FILE
# ============================================================

DATA_FILE = PROCESSED_DATA_FILE

MODEL_FILE = XGBOOST_MODEL_FILE

THRESHOLD_FILE = XGBOOST_THRESHOLD_FILE

THRESHOLD_RESULT_FILE = XGBOOST_THRESHOLD_RESULTS_FILE


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("1. ĐỌC DỮ LIỆU")
print("=" * 70)

print(
    f"File dữ liệu: {DATA_FILE}"
)

df = pd.read_csv(DATA_FILE)

print("Đọc dữ liệu thành công!")

print(
    f"Số dòng: {len(df):,}"
)

print(
    f"Số cột : {len(df.columns)}"
)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)


if "balanceDiffOrig" not in df.columns:

    df["balanceDiffOrig"] = (
        df["oldbalanceOrg"]
        - df["newbalanceOrig"]
    )

    print(
        "Đã tạo balanceDiffOrig."
    )

else:

    print(
        "balanceDiffOrig đã tồn tại."
    )


if "balanceDiffDest" not in df.columns:

    df["balanceDiffDest"] = (
        df["newbalanceDest"]
        - df["oldbalanceDest"]
    )

    print(
        "Đã tạo balanceDiffDest."
    )

else:

    print(
        "balanceDiffDest đã tồn tại."
    )


# ============================================================
# 4. CHỌN FEATURES
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


missing_columns = [
    col
    for col in FEATURES + [TARGET]
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

    print(
        " -",
        feature
    )


# ============================================================
# 5. KIỂM TRA PHÂN BỐ TARGET
# ============================================================

print("\n" + "=" * 70)
print("4. PHÂN BỐ TARGET")
print("=" * 70)


print(
    y.value_counts()
)


fraud_count = int(
    (y == 1).sum()
)

normal_count = int(
    (y == 0).sum()
)


print(
    f"Normal: {normal_count:,}"
)

print(
    f"Fraud : {fraud_count:,}"
)


# ============================================================
# 6. CHIA TRAIN / VALIDATION / TEST
# ============================================================

print("\n" + "=" * 70)
print("5. CHIA TRAIN / VALIDATION / TEST")
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
# 7. TÍNH SCALE_POS_WEIGHT
# ============================================================

print("\n" + "=" * 70)
print("6. XỬ LÝ MẤT CÂN BẰNG")
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


print(
    f"Normal trong Train: {train_normal:,}"
)

print(
    f"Fraud trong Train : {train_fraud:,}"
)

print(
    f"scale_pos_weight   : {scale_pos_weight:.2f}"
)


# ============================================================
# 8. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("7. PREPROCESSING")
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
# 9. XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("8. TẠO XGBOOST")
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
# 11. HUẤN LUYỆN
# ============================================================

print("\n" + "=" * 70)
print("9. HUẤN LUYỆN XGBOOST")
print("=" * 70)


print(
    "Đang huấn luyện..."
)

print(
    "Dataset có hơn 6 triệu dòng."
)

print(
    "Quá trình này có thể mất thời gian."
)


pipeline.fit(

    X_train,

    y_train

)


print(
    "Huấn luyện hoàn tất!"
)


# ============================================================
# 12. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("10. DỰ ĐOÁN VALIDATION")
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
print("11. TÌM THRESHOLD TRÊN VALIDATION")
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
# 14. CHỌN THRESHOLD F1 CAO NHẤT
# ============================================================

best_index = (
    results_df["f1"].idxmax()
)


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
print("12. THRESHOLD TỐT NHẤT")
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
# 15. LƯU THRESHOLD RESULTS
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
# 16. DỰ ĐOÁN TEST
# ============================================================

print("\n" + "=" * 70)
print("13. ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST")
print("=" * 70)


print(
    "Đang dự đoán Test..."
)


y_test_prob = pipeline.predict_proba(

    X_test

)[:, 1]


# SỬ DỤNG THRESHOLD
# ĐÃ CHỌN TRÊN VALIDATION

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


print("\nCONFUSION MATRIX:")

print(cm)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# 18. METRICS
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
# 19. KẾT QUẢ
# ============================================================

print("\n" + "=" * 70)
print("14. KẾT QUẢ TEST - XGBOOST")
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
print("15. CHI TIẾT CONFUSION MATRIX")
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
print("16. CLASSIFICATION REPORT")
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
print("17. LƯU MODEL")
print("=" * 70)


joblib.dump(

    pipeline,

    MODEL_FILE

)


print(
    f"Đã lưu model: {MODEL_FILE}"
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
    f"Đã lưu threshold: {THRESHOLD_FILE}"
)


# ============================================================
# 24. HOÀN TẤT
# ============================================================

print("\n" + "=" * 70)
print("HOÀN TẤT XGBOOST")
print("=" * 70)