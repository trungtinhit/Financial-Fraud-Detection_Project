# ============================================================
# PREDICT TRANSACTION - PHÁT HIỆN GIAN LẬN
# Financial-Fraud-Detection_Project
# ============================================================

import sys
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

# Thư mục scripts/
SCRIPT_DIR = Path(__file__).resolve().parent

# Thêm scripts vào sys.path để import paths.py
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paths import (
    XGBOOST_MODEL_FILE,
    XGBOOST_THRESHOLD_FILE
)


# ============================================================
# 2. CÁC FEATURE GỐC CỦA BÀI TOÁN
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


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model():
    print("=" * 60)
    print("LOAD XGBOOST MODEL")
    print("=" * 60)

    print(f"Model: {XGBOOST_MODEL_FILE}")

    if not XGBOOST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model:\n{XGBOOST_MODEL_FILE}"
        )

    model = joblib.load(XGBOOST_MODEL_FILE)

    print("Đã load model thành công.")

    return model


# ============================================================
# 4. LOAD THRESHOLD
# ============================================================

def load_threshold():

    print(f"Threshold file: {XGBOOST_THRESHOLD_FILE}")

    if not XGBOOST_THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy threshold:\n{XGBOOST_THRESHOLD_FILE}"
        )

    with open(XGBOOST_THRESHOLD_FILE, "r", encoding="utf-8") as f:
        threshold = float(f.read().strip())

    print(f"Threshold = {threshold:.2f}")

    return threshold


# ============================================================
# 5. TẠO FEATURE TỪ GIAO DỊCH
# ============================================================

def create_features(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest
):

    # --------------------------------------------------------
    # Feature chênh lệch số dư người gửi
    # --------------------------------------------------------

    balanceDiffOrig = (
        oldbalanceOrg - newbalanceOrig
    )

    # --------------------------------------------------------
    # Feature chênh lệch số dư người nhận
    # --------------------------------------------------------

    balanceDiffDest = (
        newbalanceDest - oldbalanceDest
    )

    # --------------------------------------------------------
    # Tạo DataFrame
    # --------------------------------------------------------

    data = {
        "step": [step],
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
        "balanceDiffOrig": [balanceDiffOrig],
        "balanceDiffDest": [balanceDiffDest]
    }

    df = pd.DataFrame(data)

    # Đảm bảo đúng thứ tự feature
    df = df[FEATURES]

    return df


# ============================================================
# 6. DỰ ĐOÁN
# ============================================================

def predict_transaction(model, threshold, transaction):

    # --------------------------------------------------------
    # Lấy xác suất gian lận
    # --------------------------------------------------------

    probability = model.predict_proba(transaction)[0][1]

    # --------------------------------------------------------
    # Áp dụng threshold
    # --------------------------------------------------------

    prediction = int(probability >= threshold)

    return probability, prediction


# ============================================================
# 7. HIỂN THỊ KẾT QUẢ
# ============================================================

def show_result(transaction, probability, prediction, threshold):

    print()
    print("=" * 60)
    print("KẾT QUẢ PHÁT HIỆN GIAN LẬN")
    print("=" * 60)

    print()
    print("THÔNG TIN GIAO DỊCH")
    print("-" * 60)

    print(transaction.to_string(index=False))

    print()
    print("KẾT QUẢ MODEL")
    print("-" * 60)

    print(f"Xác suất gian lận : {probability:.6f}")
    print(f"Xác suất (%)      : {probability * 100:.2f}%")
    print(f"Threshold          : {threshold:.2f}")

    print()
    print("KẾT LUẬN")
    print("-" * 60)

    if prediction == 1:

        print("⚠ CẢNH BÁO: GIAO DỊCH CÓ KHẢ NĂNG GIAN LẬN")

    else:

        print("✓ GIAO DỊCH ĐƯỢC PHÂN LOẠI LÀ BÌNH THƯỜNG")

    print()
    print("=" * 60)


# ============================================================
# 8. NHẬP GIAO DỊCH
# ============================================================

def input_transaction():

    print()
    print("=" * 60)
    print("NHẬP THÔNG TIN GIAO DỊCH")
    print("=" * 60)

    print()
    print("Các loại giao dịch:")
    print("PAYMENT")
    print("TRANSFER")
    print("CASH_OUT")
    print("CASH_IN")
    print("DEBIT")

    print()

    # --------------------------------------------------------
    # step
    # --------------------------------------------------------

    step = int(
        input("step: ")
    )

    # --------------------------------------------------------
    # type
    # --------------------------------------------------------

    transaction_type = input(
        "type: "
    ).strip().upper()

    valid_types = [
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT"
    ]

    if transaction_type not in valid_types:

        raise ValueError(
            "type không hợp lệ. "
            "Hãy sử dụng PAYMENT, TRANSFER, CASH_OUT, "
            "CASH_IN hoặc DEBIT."
        )

    # --------------------------------------------------------
    # amount
    # --------------------------------------------------------

    amount = float(
        input("amount: ")
    )

    # --------------------------------------------------------
    # Người gửi
    # --------------------------------------------------------

    oldbalanceOrg = float(
        input("oldbalanceOrg: ")
    )

    newbalanceOrig = float(
        input("newbalanceOrig: ")
    )

    # --------------------------------------------------------
    # Người nhận
    # --------------------------------------------------------

    oldbalanceDest = float(
        input("oldbalanceDest: ")
    )

    newbalanceDest = float(
        input("newbalanceDest: ")
    )

    # --------------------------------------------------------
    # Tạo DataFrame
    # --------------------------------------------------------

    transaction = create_features(
        step=step,
        transaction_type=transaction_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest
    )

    return transaction


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("HỆ THỐNG PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG")
    print("MODEL: XGBOOST")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = load_model()

        # ----------------------------------------------------
        # Load threshold
        # ----------------------------------------------------

        threshold = load_threshold()

        # ----------------------------------------------------
        # Nhập giao dịch
        # ----------------------------------------------------

        transaction = input_transaction()

        # ----------------------------------------------------
        # Dự đoán
        # ----------------------------------------------------

        probability, prediction = predict_transaction(
            model,
            threshold,
            transaction
        )

        # ----------------------------------------------------
        # Hiển thị kết quả
        # ----------------------------------------------------

        show_result(
            transaction,
            probability,
            prediction,
            threshold
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("LỖI")
        print("=" * 60)

        print(type(e).__name__)
        print(e)


# ============================================================
# 10. CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":
    main()