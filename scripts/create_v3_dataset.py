# ============================================================
# CREATE DATASET FOR XGBOOST DEPLOYMENT V3
#
# Feature mới:
# transactions_per_hour
#
# Định nghĩa:
# Số giao dịch trước đó của cùng tài khoản nameOrig
# trong cùng step (giờ PaySim).
# ============================================================

import pandas as pd

from paths import (
    RAW_DATA_FILE,
    V3_DATA_FILE,
    XGBOOST_V3_FREQUENCY_ANALYSIS_FILE
)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("=" * 70)
print("CREATE DATASET FOR DEPLOYMENT V3")
print("=" * 70)

print("\nĐọc dữ liệu raw:")
print(RAW_DATA_FILE)


columns = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "isFraud"
]


df = pd.read_csv(
    RAW_DATA_FILE,
    usecols=columns,
    dtype={
        "step": "int16",
        "amount": "float64",
        "oldbalanceOrg": "float64",
        "isFraud": "int8"
    }
)


print("\nKích thước dữ liệu:")
print(df.shape)

print("\nCác cột:")
print(df.columns.tolist())


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    df.isnull().sum()
)


# ============================================================
# 3. CREATE TRANSACTIONS PER HOUR
#
# cumcount():
#
# giao dịch đầu tiên  -> 0
# giao dịch thứ hai   -> 1
# giao dịch thứ ba    -> 2
#
# Không sử dụng dữ liệu tương lai.
# ============================================================

print(
    "\nĐang tạo transactions_per_hour..."
)


df["transactions_per_hour"] = (
    df
    .groupby(
        [
            "nameOrig",
            "step"
        ],
        sort=False
    )
    .cumcount()
    .astype("int32")
)


print(
    "Đã tạo transactions_per_hour."
)


# ============================================================
# 4. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("TRANSACTIONS PER HOUR STATISTICS")
print("=" * 70)


print(
    df["transactions_per_hour"]
    .describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)


# ============================================================
# 5. NORMAL VS FRAUD
# ============================================================

print("\n" + "=" * 70)
print("NORMAL VS FRAUD")
print("=" * 70)


class_summary = (
    df
    .groupby(
        "isFraud"
    )["transactions_per_hour"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "max"
        ]
    )
)


print(
    class_summary
)


# ============================================================
# 6. FRAUD RATE BY FREQUENCY
# ============================================================

frequency_analysis = (
    df
    .groupby(
        "transactions_per_hour"
    )
    .agg(
        total_transactions=(
            "isFraud",
            "size"
        ),

        fraud_transactions=(
            "isFraud",
            "sum"
        )
    )
    .reset_index()
)


frequency_analysis[
    "fraud_rate"
] = (
    frequency_analysis[
        "fraud_transactions"
    ]
    /
    frequency_analysis[
        "total_transactions"
    ]
)


print("\nFraud rate theo frequency:")

print(
    frequency_analysis.head(20)
)


# ============================================================
# 7. SAVE ANALYSIS
# ============================================================

XGBOOST_V3_FREQUENCY_ANALYSIS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


frequency_analysis.to_csv(
    XGBOOST_V3_FREQUENCY_ANALYSIS_FILE,
    index=False
)


print(
    "\nFrequency analysis saved:"
)

print(
    XGBOOST_V3_FREQUENCY_ANALYSIS_FILE
)


# ============================================================
# 8. CREATE FINAL V3 DATASET
#
# Không lưu nameOrig vào model dataset.
# Không sử dụng step trong V3.
# ============================================================

v3_columns = [
    "type",
    "amount",
    "oldbalanceOrg",
    "transactions_per_hour",
    "isFraud"
]


v3_df = df[
    v3_columns
].copy()


# ============================================================
# 9. SAVE V3 DATASET
# ============================================================

V3_DATA_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


v3_df.to_csv(
    V3_DATA_FILE,
    index=False
)


print("\nDataset V3 saved:")

print(
    V3_DATA_FILE
)


print("\nV3 columns:")

print(
    v3_df.columns.tolist()
)


print("\nV3 preview:")

print(
    v3_df.head(10)
)


print("\n" + "=" * 70)
print("HOÀN THÀNH CREATE V3 DATASET")
print("=" * 70)