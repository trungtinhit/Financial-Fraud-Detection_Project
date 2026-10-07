import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.config.paths import PROCESSED_DATA_FILE

input_file = PROCESSED_DATA_FILE

df = pd.read_csv(input_file)

# Tạo feature chênh lệch số dư người gửi
df["balanceDiffOrig"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# Tạo feature chênh lệch số dư người nhận
df["balanceDiffDest"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

print(df.head())
print(df.shape)


print("================================================")
print(df[
    [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "balanceDiffOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balanceDiffDest"
    ]
].describe())

print("\n=== PHAN BO TYPE ===")
print(df["type"].value_counts())

print("\n=== TYPE VA FRAUD ===")
print(pd.crosstab(df["type"], df["isFraud"]))

print("\n=== FLAGGED FRAUD ===")
print(pd.crosstab(df["isFlaggedFraud"], df["isFraud"]))