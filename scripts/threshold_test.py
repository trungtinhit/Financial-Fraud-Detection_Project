# ============================================================
# THRESHOLD TUNING - PHÁT HIỆN GIAN LẬN
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

from paths import (
    PROCESSED_DATA_FILE,
    LOGISTIC_LEGACY_MODEL_FILE,
    LOGISTIC_THRESHOLD_RESULTS_FILE
)


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

print(f"Số dòng: {len(df):,}")


# ============================================================
# 3. TẠO 2 FEATURE NẾU CHƯA CÓ
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


if "balanceDiffDest" not in df.columns:

    print("Đang tạo balanceDiffDest...")

    df["balanceDiffDest"] = (
        df["newbalanceDest"]
        - df["oldbalanceDest"]
    )


# ============================================================
# 4. CHỌN FEATURE
# ============================================================

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


X = df[FEATURES]

y = df[TARGET]


# ============================================================
# 5. CHIA TRAIN / TEST
# ============================================================

print("\n" + "=" * 70)
print("3. CHIA TRAIN / TEST")
print("=" * 70)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print(f"Train: {len(X_train):,}")
print(f"Test : {len(X_test):,}")


# ============================================================
# 6. LOAD MODEL ĐÃ TRAIN
# ============================================================

print("\n" + "=" * 70)
print("4. LOAD MODEL")
print("=" * 70)


pipeline = joblib.load(MODEL_FILE)

print("Đã load model thành công.")


# ============================================================
# 7. LẤY XÁC SUẤT DỰ ĐOÁN
# ============================================================

print("\n" + "=" * 70)
print("5. TÍNH XÁC SUẤT")
print("=" * 70)

print("Đang dự đoán xác suất trên Test...")

y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Hoàn tất.")


# ============================================================
# 8. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print(f"\nPR-AUC = {pr_auc:.6f}")


# ============================================================
# 9. TEST NHIỀU THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("6. THRESHOLD TUNING")
print("=" * 70)


thresholds = np.arange(
    0.70,
    1.00,
    0.01
)


results = []


for threshold in thresholds:

    # Nếu xác suất >= threshold
    # thì dự đoán Fraud = 1

    y_pred_threshold = (
        y_prob >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    cm = confusion_matrix(
        y_test,
        y_pred_threshold
    )


    tn, fp, fn, tp = cm.ravel()


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


# ============================================================
# 10. HIỂN THỊ KẾT QUẢ
# ============================================================

results_df = pd.DataFrame(results)


print("\n")
print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 11. TÌM THRESHOLD CÓ F1 CAO NHẤT
# ============================================================

best_index = results_df[
    "f1"
].idxmax()


best_result = results_df.loc[
    best_index
]


print("\n" + "=" * 70)
print("7. THRESHOLD CÓ F1 CAO NHẤT")
print("=" * 70)


print(
    f"Threshold : "
    f"{best_result['threshold']:.2f}"
)

print(
    f"Precision : "
    f"{best_result['precision']:.6f}"
)

print(
    f"Recall    : "
    f"{best_result['recall']:.6f}"
)

print(
    f"F1-score  : "
    f"{best_result['f1']:.6f}"
)

print(
    f"TN        : "
    f"{int(best_result['TN']):,}"
)

print(
    f"FP        : "
    f"{int(best_result['FP']):,}"
)

print(
    f"FN        : "
    f"{int(best_result['FN']):,}"
)

print(
    f"TP        : "
    f"{int(best_result['TP']):,}"
)


# ============================================================
# 12. LƯU BẢNG KẾT QUẢ
# ============================================================

OUTPUT_FILE = LOGISTIC_THRESHOLD_RESULTS_FILE

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nĐã lưu kết quả vào:")
print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("HOÀN TẤT")
print("=" * 70)