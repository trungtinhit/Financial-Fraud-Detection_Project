# ============================================================
# PHÂN TÍCH FEATURE IMPORTANCE - XGBOOST
# PaySim Fraud Detection
# ============================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from paths import (
    XGBOOST_MODEL_FILE,
    XGBOOST_FEATURE_IMPORTANCE_FILE,
    XGBOOST_FEATURE_IMPORTANCE_PLOT
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

MODEL_FILE = XGBOOST_MODEL_FILE

OUTPUT_CSV = XGBOOST_FEATURE_IMPORTANCE_FILE

OUTPUT_PNG = XGBOOST_FEATURE_IMPORTANCE_PLOT


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 70)
print("1. LOAD XGBOOST MODEL")
print("=" * 70)

pipeline = joblib.load(MODEL_FILE)

print("Đã load model:")
print(MODEL_FILE)


# ============================================================
# 3. LẤY PREPROCESSOR VÀ MODEL
# ============================================================

preprocessor = pipeline.named_steps["preprocessing"]

model = pipeline.named_steps["model"]


print("\nPreprocessor:")
print(type(preprocessor))

print("\nModel:")
print(type(model))


# ============================================================
# 4. LẤY FEATURE SAU KHI ONE-HOT ENCODING
# ============================================================

print("\n" + "=" * 70)
print("2. LẤY FEATURE SAU PREPROCESSING")
print("=" * 70)


feature_names = (
    preprocessor
    .get_feature_names_out()
)


importances = model.feature_importances_


print(
    f"Số feature sau preprocessing: "
    f"{len(feature_names)}"
)

print(
    f"Số importance của XGBoost: "
    f"{len(importances)}"
)


# ============================================================
# 5. TẠO DATAFRAME
# ============================================================

importance_df = pd.DataFrame({

    "feature": feature_names,

    "importance": importances

})


# Sắp xếp giảm dần

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 6. IN TOÀN BỘ KẾT QUẢ
# ============================================================

print("\n" + "=" * 70)
print("3. FEATURE IMPORTANCE")
print("=" * 70)


print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 7. TOP FEATURE
# ============================================================

print("\n" + "=" * 70)
print("4. TOP FEATURE")
print("=" * 70)


TOP_N = 15


top_features = (
    importance_df
    .head(TOP_N)
)


print(
    top_features.to_string(
        index=False
    )
)


# ============================================================
# 8. LƯU CSV
# ============================================================

importance_df.to_csv(

    OUTPUT_CSV,

    index=False,

    encoding="utf-8-sig"

)


print("\nĐã lưu:")
print(OUTPUT_CSV)


# ============================================================
# 9. VẼ BIỂU ĐỒ TOP 15
# ============================================================

print("\n" + "=" * 70)
print("5. VẼ BIỂU ĐỒ")
print("=" * 70)


plot_df = (
    importance_df
    .head(TOP_N)
    .sort_values(
        "importance",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(

    plot_df["feature"],

    plot_df["importance"]

)


plt.xlabel(
    "Feature Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Top 15 Feature Importance - XGBoost"
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PNG,

    dpi=300

)


plt.show()


print("\nĐã lưu biểu đồ:")
print(OUTPUT_PNG)


# ============================================================
# 10. TỔNG KẾT
# ============================================================

print("\n" + "=" * 70)
print("HOÀN TẤT FEATURE IMPORTANCE")
print("=" * 70)

print(
    f"File CSV : {OUTPUT_CSV}"
)

print(
    f"Biểu đồ  : {OUTPUT_PNG}"
)