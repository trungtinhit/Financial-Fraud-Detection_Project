# ============================================================
# PREDICT TRANSACTION - DEPLOYMENT MODEL
# Financial Fraud Detection
#
# FEATURES:
# - step
# - type
# - amount
# - oldbalanceOrg
#
# MODEL:
# - xgboost_deployment_model.pkl
#
# THRESHOLD:
# - xgboost_deployment_threshold.txt
# ============================================================

import sys
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. IMPORT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from paths import (
    XGBOOST_DEPLOYMENT_MODEL_FILE,
    XGBOOST_DEPLOYMENT_THRESHOLD_FILE
)


# ============================================================
# 2. CẤU HÌNH
# ============================================================

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg"
]


VALID_TYPES = [
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "DEBIT"
]


OUTGOING_TYPES = [
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "DEBIT"
]


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model():

    print("=" * 65)
    print("LOAD XGBOOST DEPLOYMENT MODEL")
    print("=" * 65)

    if not XGBOOST_DEPLOYMENT_MODEL_FILE.exists():

        raise FileNotFoundError(
            "Không tìm thấy deployment model:\n"
            f"{XGBOOST_DEPLOYMENT_MODEL_FILE}"
        )

    model = joblib.load(
        XGBOOST_DEPLOYMENT_MODEL_FILE
    )

    print(
        f"Model: {XGBOOST_DEPLOYMENT_MODEL_FILE}"
    )

    print(
        "Đã load deployment model thành công."
    )

    return model


# ============================================================
# 4. LOAD THRESHOLD
# ============================================================

def load_threshold():

    if not XGBOOST_DEPLOYMENT_THRESHOLD_FILE.exists():

        raise FileNotFoundError(
            "Không tìm thấy deployment threshold:\n"
            f"{XGBOOST_DEPLOYMENT_THRESHOLD_FILE}"
        )

    with open(
        XGBOOST_DEPLOYMENT_THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        threshold = float(
            f.read().strip()
        )

    print(
        f"Threshold file: "
        f"{XGBOOST_DEPLOYMENT_THRESHOLD_FILE}"
    )

    print(
        f"Threshold = {threshold:.2f}"
    )

    return threshold


# ============================================================
# 5. VALIDATE TRANSACTION
# ============================================================

def validate_transaction(
    step,
    transaction_type,
    amount,
    oldbalanceOrg
):

    errors = []
    warnings = []


    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    if step < 1:

        errors.append(
            "step phải lớn hơn hoặc bằng 1."
        )


    # --------------------------------------------------------
    # TYPE
    # --------------------------------------------------------

    if transaction_type not in VALID_TYPES:

        errors.append(
            "Loại giao dịch không hợp lệ."
        )


    # --------------------------------------------------------
    # AMOUNT
    # --------------------------------------------------------

    if amount <= 0:

        errors.append(
            "Số tiền giao dịch phải lớn hơn 0."
        )


    # --------------------------------------------------------
    # BALANCE
    # --------------------------------------------------------

    if oldbalanceOrg < 0:

        errors.append(
            "Số dư tài khoản không được âm."
        )


    # --------------------------------------------------------
    # BUSINESS RULE:
    # outgoing transaction > available balance
    # --------------------------------------------------------

    if (
        transaction_type in OUTGOING_TYPES
        and amount > oldbalanceOrg
    ):

        warnings.append(
            "Số tiền giao dịch lớn hơn "
            "số dư hiện tại của tài khoản."
        )


    # --------------------------------------------------------
    # BALANCE = 0
    # --------------------------------------------------------

    if (
        transaction_type in OUTGOING_TYPES
        and oldbalanceOrg == 0
    ):

        warnings.append(
            "Tài khoản gửi có số dư bằng 0."
        )


    # --------------------------------------------------------
    # FULL BALANCE TRANSACTION
    # --------------------------------------------------------

    if (
        transaction_type in OUTGOING_TYPES
        and oldbalanceOrg > 0
        and abs(amount - oldbalanceOrg) < 0.01
    ):

        warnings.append(
            "Giao dịch sử dụng toàn bộ số dư tài khoản."
        )


    return errors, warnings


# ============================================================
# 6. CREATE MODEL INPUT
# ============================================================

def create_transaction_dataframe(
    step,
    transaction_type,
    amount,
    oldbalanceOrg
):

    transaction = pd.DataFrame(
        {
            "step": [step],
            "type": [transaction_type],
            "amount": [amount],
            "oldbalanceOrg": [oldbalanceOrg]
        }
    )

    # Đảm bảo đúng thứ tự feature
    transaction = transaction[FEATURES]

    return transaction


# ============================================================
# 7. PREDICT
# ============================================================

def predict_transaction(
    model,
    threshold,
    transaction
):

    probability = (
        model
        .predict_proba(transaction)[0][1]
    )

    prediction = int(
        probability >= threshold
    )

    return probability, prediction


# ============================================================
# 8. INPUT
# ============================================================

def input_transaction():

    print()
    print("=" * 65)
    print("NHẬP THÔNG TIN GIAO DỊCH")
    print("=" * 65)

    print()
    print("Các loại giao dịch:")

    for transaction_type in VALID_TYPES:

        print(
            f"- {transaction_type}"
        )

    print()

    step = int(
        input("step: ")
    )

    transaction_type = (
        input("type: ")
        .strip()
        .upper()
    )

    amount = float(
        input("amount: ")
    )

    oldbalanceOrg = float(
        input("oldbalanceOrg: ")
    )

    return {
        "step": step,
        "transaction_type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg
    }


# ============================================================
# 9. SHOW RESULT
# ============================================================

def show_result(
    transaction,
    probability,
    prediction,
    threshold,
    warnings
):

    print()
    print("=" * 65)
    print("KẾT QUẢ PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG")
    print("=" * 65)


    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    print()
    print("THÔNG TIN GIAO DỊCH")
    print("-" * 65)

    print(
        transaction.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # BUSINESS WARNINGS
    # --------------------------------------------------------

    if warnings:

        print()
        print("CẢNH BÁO NGHIỆP VỤ")
        print("-" * 65)

        for warning in warnings:

            print(
                f"⚠ {warning}"
            )


    # --------------------------------------------------------
    # MODEL RESULT
    # --------------------------------------------------------

    print()
    print("KẾT QUẢ MODEL")
    print("-" * 65)

    print(
        f"Xác suất gian lận : "
        f"{probability:.6f}"
    )

    print(
        f"Xác suất (%)      : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Threshold          : "
        f"{threshold:.2f}"
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("KẾT LUẬN")
    print("-" * 65)

    if prediction == 1:

        print(
            "⚠ MODEL CẢNH BÁO: "
            "GIAO DỊCH CÓ KHẢ NĂNG GIAN LẬN"
        )

    else:

        print(
            "✓ MODEL KHÔNG PHÁT HIỆN "
            "DẤU HIỆU GIAN LẬN"
        )


    if warnings:

        print()
        print(
            "⚠ LƯU Ý: Giao dịch có ít nhất một "
            "điểm bất thường về nghiệp vụ."
        )

        print(
            "Kết quả model và kiểm tra nghiệp vụ "
            "nên được xem xét đồng thời."
        )


    print()
    print("=" * 65)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("HỆ THỐNG PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG")
    print("DEPLOYMENT MODEL: XGBOOST")
    print("=" * 65)

    try:

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = load_model()


        # ----------------------------------------------------
        # LOAD THRESHOLD
        # ----------------------------------------------------

        threshold = load_threshold()


        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        values = input_transaction()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        errors, warnings = validate_transaction(
            **values
        )


        # ----------------------------------------------------
        # HARD ERRORS
        # ----------------------------------------------------

        if errors:

            print()
            print("=" * 65)
            print("DỮ LIỆU GIAO DỊCH KHÔNG HỢP LỆ")
            print("=" * 65)

            for error in errors:

                print(
                    f"✗ {error}"
                )

            return


        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        transaction = (
            create_transaction_dataframe(
                **values
            )
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        probability, prediction = (
            predict_transaction(
                model,
                threshold,
                transaction
            )
        )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        show_result(
            transaction,
            probability,
            prediction,
            threshold,
            warnings
        )


    except ValueError as e:

        print()
        print("=" * 65)
        print("LỖI DỮ LIỆU")
        print("=" * 65)

        print(e)


    except Exception as e:

        print()
        print("=" * 65)
        print("LỖI HỆ THỐNG")
        print("=" * 65)

        print(
            type(e).__name__
        )

        print(e)


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":

    main()