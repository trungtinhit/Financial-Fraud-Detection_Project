# ============================================================
# STREAMLIT APP - OPTIMIZED
# Financial Fraud Detection
# ============================================================

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# 1. PATH CONFIG
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SERVICES_DIR = APP_DIR / "services"

if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))


from fraud_service import (
    VALID_TYPES,
    load_model,
    load_threshold,
    analyze_transaction
)


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# 3. HISTORY FILE
# ============================================================

HISTORY_DIR = PROJECT_ROOT / "results" / "app"

HISTORY_FILE = (
    HISTORY_DIR /
    "transaction_history.csv"
)

HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_threshold():
    return load_threshold()


try:

    model = get_model()
    threshold = get_threshold()

except Exception as e:

    st.error(
        f"Không thể load model hoặc threshold: {e}"
    )

    st.stop()


# ============================================================
# 5. STEP TỰ ĐỘNG
# ============================================================

def get_current_step():

    # PaySim dùng step theo giờ.
    # Bản demo dùng số giờ tính từ đầu năm.

    now = datetime.now()

    start_of_year = datetime(
        now.year,
        1,
        1
    )

    hours = int(
        (
            now - start_of_year
        ).total_seconds()
        // 3600
    )

    return max(
        hours,
        1
    )


# ============================================================
# 6. RISK LEVEL
# ============================================================

def get_risk_level(
    probability,
    threshold
):

    # --------------------------------------------------------
    # FRAUD
    # --------------------------------------------------------

    if probability >= threshold:

        return {
            "level": "CẢNH BÁO GIAN LẬN",
            "status": "fraud"
        }


    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if probability >= 0.90:

        return {
            "level": "RỦI RO CAO",
            "status": "high"
        }


    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    if probability >= 0.70:

        return {
            "level": "CẦN CHÚ Ý",
            "status": "medium"
        }


    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    return {
        "level": "RỦI RO THẤP",
        "status": "low"
    }


# ============================================================
# 7. SAVE HISTORY
# ============================================================

def save_history(
    transaction_type,
    amount,
    oldbalanceOrg,
    step,
    probability,
    threshold,
    risk_level,
    prediction,
    warnings
):

    record = pd.DataFrame(
        [
            {
                "timestamp":
                    datetime.now()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "type":
                    transaction_type,

                "amount":
                    amount,

                "oldbalanceOrg":
                    oldbalanceOrg,

                "step":
                    step,

                "probability":
                    probability,

                "probability_percent":
                    probability * 100,

                "threshold":
                    threshold,

                "risk_level":
                    risk_level,

                "prediction":
                    prediction,

                "warnings":
                    " | ".join(
                        warnings
                    )
            }
        ]
    )


    if HISTORY_FILE.exists():

        old_history = pd.read_csv(
            HISTORY_FILE
        )

        history = pd.concat(
            [
                old_history,
                record
            ],
            ignore_index=True
        )

    else:

        history = record


    history.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 8. HEADER
# ============================================================

st.title(
    "🔍 Hệ thống phát hiện giao dịch bất thường"
)

st.caption(
    "Financial Fraud Detection "
    "sử dụng XGBoost Deployment Model"
)


# ============================================================
# 9. MODEL INFO
# ============================================================

with st.expander(
    "Thông tin mô hình",
    expanded=False
):

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Model",
            "XGBoost"
        )


    with col2:

        st.metric(
            "Số feature",
            "4"
        )


    with col3:

        st.metric(
            "Threshold",
            f"{threshold * 100:.0f}%"
        )


    st.markdown(
        """
        **Feature sử dụng**

        - `step`
        - `type`
        - `amount`
        - `oldbalanceOrg`
        """
    )


    st.info(
        "Model triển khai không cần "
        "số dư người nhận hoặc "
        "số dư sau giao dịch."
    )


# ============================================================
# 10. INPUT FORM
# ============================================================

st.subheader(
    "Thông tin giao dịch"
)


with st.form(
    "transaction_form"
):

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    with col1:

        transaction_type = st.selectbox(
            "Loại giao dịch",
            VALID_TYPES
        )


        amount = st.number_input(
            "Số tiền giao dịch",
            min_value=0.0,
            value=1000000.0,
            step=100000.0,
            format="%.2f"
        )


    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    with col2:

        oldbalanceOrg = st.number_input(
            "Số dư hiện tại của tài khoản",
            min_value=0.0,
            value=5000000.0,
            step=100000.0,
            format="%.2f"
        )


        st.write(
            "Mốc thời gian"
        )

        st.info(
            "Step được hệ thống "
            "tự động xác định."
        )


    submitted = st.form_submit_button(
        "🔎 Kiểm tra giao dịch",
        use_container_width=True
    )


# ============================================================
# 11. PROCESS
# ============================================================

if submitted:

    step = get_current_step()


    result = analyze_transaction(
        step=step,
        transaction_type=transaction_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        model=model,
        threshold=threshold
    )


    # ========================================================
    # ERROR
    # ========================================================

    if not result["success"]:

        st.divider()

        st.subheader(
            "Kết quả kiểm tra"
        )


        st.error(
            "Dữ liệu giao dịch không hợp lệ."
        )


        for error in result["errors"]:

            st.error(
                f"❌ {error}"
            )


        st.stop()


    # ========================================================
    # RESULT DATA
    # ========================================================

    probability = result[
        "probability"
    ]

    prediction = result[
        "prediction"
    ]

    warnings = result[
        "warnings"
    ]

    transaction = result[
        "transaction"
    ]


    risk = get_risk_level(
        probability,
        threshold
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    save_history(
        transaction_type=transaction_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        step=step,
        probability=probability,
        threshold=threshold,
        risk_level=risk["level"],
        prediction=prediction,
        warnings=warnings
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    st.divider()

    st.subheader(
        "Kết quả phân tích"
    )


    # ========================================================
    # BUSINESS WARNINGS
    # ========================================================

    if warnings:

        st.warning(
            "Phát hiện điểm cần chú ý "
            "về nghiệp vụ."
        )


        for warning in warnings:

            st.write(
                f"⚠️ {warning}"
            )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Xác suất gian lận",
            f"{probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Threshold",
            f"{threshold * 100:.0f}%"
        )


    with col3:

        st.metric(
            "Mức độ rủi ro",
            risk["level"]
        )


    # ========================================================
    # RISK BAR
    # ========================================================

    st.write(
        "### Mức độ rủi ro"
    )


    st.progress(
        min(
            max(
                float(probability),
                0.0
            ),
            1.0
        )
    )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    if risk["status"] == "fraud":

        st.error(
            "🚨 Model cảnh báo giao dịch "
            "có khả năng gian lận."
        )


    elif risk["status"] == "high":

        st.warning(
            "⚠️ Giao dịch có mức rủi ro cao. "
            "Xác suất chưa vượt threshold "
            "nhưng nên được kiểm tra thủ công."
        )


    elif risk["status"] == "medium":

        st.warning(
            "⚠️ Giao dịch cần chú ý. "
            "Nên theo dõi thêm trước khi xử lý."
        )


    else:

        st.success(
            "✅ Giao dịch có mức rủi ro thấp."
        )


    # ========================================================
    # BUSINESS WARNING
    # ========================================================

    if warnings:

        st.info(
            "Kết quả model và cảnh báo "
            "nghiệp vụ nên được xem xét "
            "đồng thời."
        )


    # ========================================================
    # TRANSACTION INPUT
    # ========================================================

    with st.expander(
        "Xem dữ liệu được đưa vào model"
    ):

        st.dataframe(
            transaction,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "Chi tiết kỹ thuật"
    ):

        st.write(
            f"Probability: "
            f"`{probability:.6f}`"
        )

        st.write(
            f"Threshold: "
            f"`{threshold:.6f}`"
        )

        st.write(
            f"Prediction: "
            f"`{prediction}`"
        )

        st.write(
            f"Step: "
            f"`{step}`"
        )


# ============================================================
# 12. HISTORY
# ============================================================

st.divider()

st.subheader(
    "Lịch sử kiểm tra giao dịch"
)


if HISTORY_FILE.exists():

    history_df = pd.read_csv(
        HISTORY_FILE
    )


    # --------------------------------------------------------
    # HIỂN THỊ 20 GIAO DỊCH GẦN NHẤT
    # --------------------------------------------------------

    history_display = (
        history_df
        .tail(20)
        .iloc[::-1]
    )


    st.dataframe(
        history_display[
            [
                "timestamp",
                "type",
                "amount",
                "oldbalanceOrg",
                "probability_percent",
                "risk_level"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_transactions = len(
        history_df
    )


    fraud_count = int(
        (
            history_df[
                "prediction"
            ] == 1
        ).sum()
    )


    high_risk_count = int(
        history_df[
            "risk_level"
        ]
        .isin(
            [
                "RỦI RO CAO",
                "CẢNH BÁO GIAN LẬN"
            ]
        )
        .sum()
    )


    st.write(
        "### Thống kê nhanh"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Tổng giao dịch",
            total_transactions
        )


    with col2:

        st.metric(
            "Cảnh báo gian lận",
            fraud_count
        )


    with col3:

        st.metric(
            "Rủi ro cao",
            high_risk_count
        )


else:

    st.info(
        "Chưa có lịch sử giao dịch."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()


st.caption(
    "Model được huấn luyện trên dữ liệu PaySim. "
    "Kết quả chỉ phản ánh dự đoán của mô hình "
    "và không phải kết luận tuyệt đối về "
    "một giao dịch thực tế."
)