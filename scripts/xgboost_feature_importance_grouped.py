# ============================================================
# GỘP FEATURE IMPORTANCE XGBOOST
# 13 FEATURE SAU ONE-HOT -> 9 FEATURE GỐC
# ============================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from paths import (
    XGBOOST_MODEL_FILE,
    XGBOOST_FEATURE_IMPORTANCE_GROUPED_FILE,
    XGBOOST_FEATURE_IMPORTANCE_GROUPED_PLOT
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

MODEL_FILE = XGBOOST_MODEL_FILE

OUTPUT_CSV = XGBOOST_FEATURE_IMPORTANCE_GROUPED_FILE

OUTPUT_PNG = XGBOOST_FEATURE_IMPORTANCE_GROUPED_PLOT


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 70)
print("1. LOAD XGBOOST MODEL")
print("=" * 70)

pipeline = joblib.load(MODEL_FILE)

print("Đã load:")
print(MODEL_FILE)


# ============================================================
# 3. LẤY PREPROCESSOR VÀ XGBOOST
# ============================================================

preprocessor = pipeline.named_steps["preprocessing"]

model = pipeline.named_steps["model"]


# ============================================================
# 4. LẤY FEATURE SAU PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("2. LẤY FEATURE SAU PREPROCESSING")
print("=" * 70)

feature_names = preprocessor.get_feature_names_out()

importances = model.feature_importances_

print("Số feature sau preprocessing:", len(feature_names))
print("Số feature importance:", len(importances))


# ============================================================
# 5. TẠO DATAFRAME
# ============================================================

df_importance = pd.DataFrame({

    "feature": feature_names,

    "importance": importances

})


# ============================================================
# 6. CHUẨN HÓA TÊN FEATURE
# ============================================================

# Ví dụ:
#
# numeric__amount
#        ↓
# amount
#
# categorical__type_PAYMENT
#        ↓
# type
#
# numeric__balanceDiffOrig
#        ↓
# balanceDiffOrig


def get_original_feature(feature_name):

    # Feature numeric
    if feature_name.startswith("numeric__"):

        return feature_name.replace(
            "numeric__",
            ""
        )

    # Feature categorical
    if feature_name.startswith("categorical__"):

        feature_name = feature_name.replace(
            "categorical__",
            ""
        )

        # Tất cả type_xxx -> type
        if feature_name.startswith("type_"):

            return "type"

        return feature_name

    # Trường hợp khác
    return feature_name


df_importance["original_feature"] = (
    df_importance["feature"]
    .apply(get_original_feature)
)


# ============================================================
# 7. GỘP IMPORTANCE
# ============================================================

grouped_df = (

    df_importance

    .groupby(
        "original_feature",
        as_index=False
    )["importance"]

    .sum()

)


# ============================================================
# 8. SẮP XẾP GIẢM DẦN
# ============================================================

grouped_df = (

    grouped_df

    .sort_values(
        "importance",
        ascending=False
    )

    .reset_index(drop=True)

)


# ============================================================
# 9. ĐỔI TÊN CỘT
# ============================================================

grouped_df = grouped_df.rename(

    columns={
        "original_feature": "feature"
    }

)


# ============================================================
# 10. KIỂM TRA 9 FEATURE
# ============================================================

expected_features = [

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


print("\n" + "=" * 70)
print("3. FEATURE IMPORTANCE SAU KHI GỘP")
print("=" * 70)

print(
    grouped_df.to_string(
        index=False
    )
)


print(
    "\nSố feature gốc:",
    len(grouped_df)
)


# ============================================================
# 11. KIỂM TRA FEATURE CÓ THIẾU KHÔNG
# ============================================================

found_features = set(
    grouped_df["feature"]
)

missing_features = (
    set(expected_features)
    - found_features
)

extra_features = (
    found_features
    - set(expected_features)
)


if missing_features:

    print("\nCẢNH BÁO - THIẾU FEATURE:")

    print(
        sorted(missing_features)
    )

else:

    print(
        "\nĐã tìm thấy đầy đủ 9 feature gốc."
    )


if extra_features:

    print("\nFeature ngoài danh sách dự kiến:")

    print(
        sorted(extra_features)
    )


# ============================================================
# 12. KIỂM TRA TỔNG IMPORTANCE
# ============================================================

total_importance = (
    grouped_df["importance"]
    .sum()
)


print("\nTổng Feature Importance:")

print(
    f"{total_importance:.6f}"
)


# ============================================================
# 13. THÊM PHẦN TRĂM
# ============================================================

grouped_df["percentage"] = (

    grouped_df["importance"]

    / total_importance

    * 100

)


# ============================================================
# 14. IN BẢNG CUỐI CÙNG
# ============================================================

print("\n" + "=" * 70)
print("4. BẢNG FEATURE IMPORTANCE CUỐI CÙNG")
print("=" * 70)


print(

    grouped_df[
        [
            "feature",
            "importance",
            "percentage"
        ]
    ]

    .to_string(
        index=False,

        formatters={

            "importance":
                "{:.6f}".format,

            "percentage":
                "{:.2f}%".format

        }

    )

)


# ============================================================
# 15. LƯU CSV
# ============================================================

grouped_df.to_csv(

    OUTPUT_CSV,

    index=False,

    encoding="utf-8-sig"

)


print("\nĐã lưu file:")

print(
    OUTPUT_CSV
)


# ============================================================
# 16. CHUẨN BỊ DỮ LIỆU VẼ
# ============================================================

plot_df = (

    grouped_df

    .sort_values(
        "importance",
        ascending=True
    )

)


# ============================================================
# 17. VẼ BIỂU ĐỒ
# ============================================================

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
    "Feature Importance - XGBoost (9 Features Gốc)"
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PNG,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ============================================================
# 18. HOÀN TẤT
# ============================================================

print("\n" + "=" * 70)
print("HOÀN TẤT")
print("=" * 70)

print(
    "CSV    :",
    OUTPUT_CSV
)

print(
    "Biểu đồ:",
    OUTPUT_PNG
)