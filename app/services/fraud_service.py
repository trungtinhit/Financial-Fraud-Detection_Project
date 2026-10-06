# ============================================================
# FRAUD SERVICE - XGBOOST DEPLOYMENT V3
#
# Features:
# - type
# - amount
# - oldbalanceOrg
# - transactions_per_hour
# ============================================================

import sys
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. PATH CONFIG
# ============================================================

SERVICE_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICE_DIR.parent
PROJECT_ROOT = APP_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS_DIR)
    )


from paths import (
    XGBOOST_DEPLOYMENT_V3_MODEL_FILE,
    XGBOOST_DEPLOYMENT_V3_THRESHOLD_FILE
)


# ============================================================
# 2. CONSTANTS
# ============================================================

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "transactions_per_hour"
]


VALID_TYPES = [
    "TRANSFER",
    "CASH_OUT",
    "PAYMENT",
    "CASH_IN",
    "DEBIT"
]


OUTGOING_TYPES = [
    "TRANSFER",
    "CASH_OUT",
    "PAYMENT",
    "DEBIT"
]


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model():

    if not XGBOOST_DEPLOYMENT_V3_MODEL_FILE.exists():

        raise FileNotFoundError(
            "Không tìm thấy XGBoost Deployment V3 model:\n"
            f"{XGBOOST_DEPLOYMENT_V3_MODEL_FILE}"
        )

    return joblib.load(
        XGBOOST_DEPLOYMENT_V3_MODEL_FILE
    )


# ============================================================
# 4. LOAD THRESHOLD
# ============================================================

def load_threshold():

    if not XGBOOST_DEPLOYMENT_V3_THRESHOLD_FILE.exists():

        raise FileNotFoundError(
            "Không tìm thấy threshold V3:\n"
            f"{XGBOOST_DEPLOYMENT_V3_THRESHOLD_FILE}"
        )

    with open(
        XGBOOST_DEPLOYMENT_V3_THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        threshold = float(
            f.read().strip()
        )

    return threshold


# ============================================================
# 5. VALIDATE TRANSACTION
# ============================================================

def validate_transaction(
    transaction_type,
    amount,
    oldbalanceOrg,
    transactions_per_hour
):

    errors = []
    warnings = []


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
    # FREQUENCY
    # --------------------------------------------------------

    if transactions_per_hour < 0:

        errors.append(
            "Số giao dịch trong giờ không được âm."
        )


    # --------------------------------------------------------
    # AMOUNT > BALANCE
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
            "Tài khoản hiện có số dư bằng 0."
        )


    # --------------------------------------------------------
    # FULL BALANCE
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

def create_model_input(
    transaction_type,
    amount,
    oldbalanceOrg,
    transactions_per_hour
):

    transaction = pd.DataFrame(
        {
            "type": [
                transaction_type
            ],

            "amount": [
                amount
            ],

            "oldbalanceOrg": [
                oldbalanceOrg
            ],

            "transactions_per_hour": [
                transactions_per_hour
            ]
        }
    )


    return transaction[
        FEATURES
    ]


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
        .predict_proba(
            transaction
        )[0][1]
    )


    prediction = int(
        probability >= threshold
    )


    return probability, prediction


# ============================================================
# 8. FULL ANALYSIS
# ============================================================

def analyze_transaction(
    transaction_type,
    amount,
    oldbalanceOrg,
    transactions_per_hour,
    model,
    threshold
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    errors, warnings = (
        validate_transaction(
            transaction_type=transaction_type,
            amount=amount,
            oldbalanceOrg=oldbalanceOrg,
            transactions_per_hour=transactions_per_hour
        )
    )


    # --------------------------------------------------------
    # STOP IF INVALID
    # --------------------------------------------------------

    if errors:

        return {
            "success": False,
            "errors": errors,
            "warnings": warnings,
            "probability": None,
            "prediction": None,
            "transaction": None
        }


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    transaction = create_model_input(
        transaction_type=transaction_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        transactions_per_hour=transactions_per_hour
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    probability, prediction = (
        predict_transaction(
            model=model,
            threshold=threshold,
            transaction=transaction
        )
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "success": True,

        "errors": [],

        "warnings": warnings,

        "probability":
            probability,

        "probability_percent":
            probability * 100,

        "prediction":
            prediction,

        "is_fraud":
            prediction == 1,

        "threshold":
            threshold,

        "transaction":
            transaction
    }