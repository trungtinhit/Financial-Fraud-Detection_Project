# ============================================================
# SO SÁNH 3 MÔ HÌNH PHÁT HIỆN GIAN LẬN
# Logistic Regression - Random Forest - XGBoost
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from paths import (
    MODEL_COMPARISON_RESULTS_FILE,
    MODEL_COMPARISON_METRICS_PLOT,
    MODEL_COMPARISON_FP_FN_PLOT
)


# ============================================================
# 1. DỮ LIỆU KẾT QUẢ TEST
# ============================================================

results = {

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Threshold": [
        0.99,
        0.94,
        0.99
    ],

    "Accuracy": [
        0.998032,
        0.999664,
        0.999772
    ],

    "Precision": [
        0.348924,
        0.948768,
        0.959239
    ],

    "Recall": [
        0.605519,
        0.781656,
        0.859578
    ],

    "F1-score": [
        0.442730,
        0.857143,
        0.906678
    ],

    "ROC-AUC": [
        0.988574,
        0.999387,
        0.999693
    ],

    "PR-AUC": [
        0.556385,
        0.919214,
        0.964059
    ],

    "TN": [
        951769,
        953109,
        953116
    ],

    "FP": [
        1392,
        52,
        45
    ],

    "FN": [
        486,
        269,
        173
    ],

    "TP": [
        746,
        963,
        1059
    ]
}


# ============================================================
# 2. TẠO DATAFRAME
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# 3. IN BẢNG RA TERMINAL
# ============================================================

print("=" * 90)
print("BẢNG SO SÁNH 3 MÔ HÌNH")
print("=" * 90)

print()

print(
    df[
        [
            "Model",
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "PR-AUC"
        ]
    ].to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Accuracy": "{:.6f}".format,
            "Precision": "{:.6f}".format,
            "Recall": "{:.6f}".format,
            "F1-score": "{:.6f}".format,
            "ROC-AUC": "{:.6f}".format,
            "PR-AUC": "{:.6f}".format
        }
    )
)


# ============================================================
# 4. IN CONFUSION MATRIX
# ============================================================

print()
print("=" * 90)
print("CONFUSION MATRIX - 3 MÔ HÌNH")
print("=" * 90)

for _, row in df.iterrows():

    print()
    print(row["Model"])

    print(
        f"TN = {int(row['TN']):,}"
    )

    print(
        f"FP = {int(row['FP']):,}"
    )

    print(
        f"FN = {int(row['FN']):,}"
    )

    print(
        f"TP = {int(row['TP']):,}"
    )


# ============================================================
# 5. LƯU BẢNG CSV
# ============================================================

output_csv = MODEL_COMPARISON_RESULTS_FILE

df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 90)
print("ĐÃ LƯU BẢNG")
print("=" * 90)

print(output_csv)


# ============================================================
# 6. CHUẨN BỊ DỮ LIỆU CHO BIỂU ĐỒ
# ============================================================

metrics = [
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "PR-AUC"
]

models = df["Model"].tolist()

x = np.arange(len(metrics))

width = 0.25


# ============================================================
# 7. BIỂU ĐỒ SO SÁNH CÁC CHỈ SỐ
# ============================================================

plt.figure(figsize=(12, 7))

for i, model in enumerate(models):

    values = df.loc[
        i,
        metrics
    ].values.astype(float)

    plt.bar(
        x + (i - 1) * width,
        values,
        width,
        label=model
    )


plt.xticks(
    x,
    metrics
)

plt.ylim(
    0,
    1.05
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Evaluation Metrics"
)

plt.title(
    "So sánh hiệu năng 3 mô hình phát hiện gian lận"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 8. LƯU BIỂU ĐỒ
# ============================================================

output_png = MODEL_COMPARISON_METRICS_PLOT

plt.savefig(
    output_png,
    dpi=300,
    bbox_inches="tight"
)

print()
print("Đã lưu biểu đồ:")

print(output_png)


plt.show()


# ============================================================
# 9. BIỂU ĐỒ FP / FN
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(models))

width = 0.35


plt.bar(
    x - width / 2,
    df["FP"],
    width,
    label="False Positive (FP)"
)


plt.bar(
    x + width / 2,
    df["FN"],
    width,
    label="False Negative (FN)"
)


plt.xticks(
    x,
    models
)

plt.ylabel(
    "Number of Transactions"
)

plt.xlabel(
    "Model"
)

plt.title(
    "So sánh False Positive và False Negative"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 10. LƯU BIỂU ĐỒ FP/FN
# ============================================================

error_png = MODEL_COMPARISON_FP_FN_PLOT

plt.savefig(
    error_png,
    dpi=300,
    bbox_inches="tight"
)

print()
print("Đã lưu biểu đồ:")

print(error_png)


plt.show()


# ============================================================
# 11. HOÀN TẤT
# ============================================================

print()
print("=" * 90)
print("HOÀN TẤT SO SÁNH 3 MÔ HÌNH")
print("=" * 90)

print()
print("Các file được tạo:")

print("1.", output_csv)

print("2.", output_png)

print("3.", error_png)