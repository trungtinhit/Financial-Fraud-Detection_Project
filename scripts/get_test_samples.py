import pandas as pd

from paths import PROCESSED_DATA_FILE

df = pd.read_csv(PROCESSED_DATA_FILE)

fraud_samples = df[df["isFraud"] == 1].sample(
    n=5,
    random_state=42
)

normal_samples = df[df["isFraud"] == 0].sample(
    n=5,
    random_state=42
)

columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]

print("=" * 80)
print("5 GIAO DỊCH FRAUD THẬT TRONG PAYSIM")
print("=" * 80)

print(
    fraud_samples[columns].to_string(index=False)
)

print("\n" + "=" * 80)
print("5 GIAO DỊCH NORMAL THẬT TRONG PAYSIM")
print("=" * 80)

print(
    normal_samples[columns].to_string(index=False)
)