# ============================================================
# STREAMLIT APP - DEPLOYMENT V2
# Financial Fraud Detection
#
# Features:
# - type
# - amount
# - oldbalanceOrg
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
    sys.path.insert(
        0,
        str(SERVICES_DIR)
    )


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
# 3. HISTORY CONFIG
# ============================================================

HISTORY_DIR = (
    PROJECT_ROOT /
    "results" /
    "app"
)

HISTORY_FILE = (
    HISTORY_DIR /
    "transaction_history_v2.csv"
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
# 5. RISK LEVEL
# ============================================================

def get_risk_level(
    probability,
    threshold
):

    if probability >= threshold:

        return (
            "CẢNH BÁO NGUY CƠ GIAN LẬN",
            "fraud"
        )

    elif probability >= 0.90:

        return (
            "RỦI RO CAO",
            "high"
        )

    elif probability >= 0.70:

        return (
            "CẦN CHÚ Ý",
            "medium"
        )

    else:

        return (
            "RỦI RO THẤP",
            "low"
        )


# ============================================================
# 6. SAVE HISTORY
# ============================================================

def save_history(
    transaction_type,
    amount,
    oldbalanceOrg,
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
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "type":
                    transaction_type,

                "amount":
                    amount,

                "oldbalanceOrg":
                    oldbalanceOrg,

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
# 7. LOAD HISTORY
# ============================================================

def load_history():

    if not HISTORY_FILE.exists():

        return pd.DataFrame()

    return pd.read_csv(
        HISTORY_FILE
    )


# ============================================================
# 8. HEADER
# ============================================================

st.title(
    "🔍 Hệ thống phát hiện giao dịch bất thường"
)

st.caption(
    "Financial Fraud Detection - "
    "XGBoost Deployment Model V2"
)


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Thông tin hệ thống"
    )

    st.write(
        "**Model:** XGBoost V2"
    )

    st.write(
        "**Số feature:** 3"
    )

    st.write(
        f"**Ngưỡng cảnh báo:** "
        f"{threshold * 100:.0f}%"
    )


    st.divider()


    st.write(
        "**Feature triển khai:**"
    )

    st.code(
        """
type
amount
oldbalanceOrg
        """
    )


    st.info(
        "Model V2 không sử dụng step, "
        "số dư người nhận hoặc "
        "thông tin sau giao dịch."
    )


# ============================================================
# 10. TABS
# ============================================================

tab_check, tab_dashboard, tab_history, tab_model = st.tabs(
    [
        "🔎 Kiểm tra giao dịch",
        "📊 Dashboard",
        "📜 Lịch sử",
        "🧠 Thông tin model"
    ]
)


# ============================================================
# TAB 1 - TRANSACTION CHECK
# ============================================================

with tab_check:

    st.subheader(
        "Thông tin giao dịch"
    )


    with st.form(
        "transaction_form"
    ):

        col1, col2 = st.columns(2)


        # ====================================================
        # LEFT
        # ====================================================

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


        # ====================================================
        # RIGHT
        # ====================================================

        with col2:

            oldbalanceOrg = st.number_input(
                "Số dư hiện tại của tài khoản",
                min_value=0.0,
                value=5000000.0,
                step=100000.0,
                format="%.2f"
            )


            st.info(
                "Model V2 chỉ sử dụng "
                "thông tin có thể biết "
                "trước khi giao dịch hoàn tất."
            )


        submitted = st.form_submit_button(
            "🔎 Kiểm tra giao dịch",
            use_container_width=True
        )


    # ========================================================
    # PROCESS
    # ========================================================

    if submitted:

        result = analyze_transaction(
            transaction_type=transaction_type,
            amount=amount,
            oldbalanceOrg=oldbalanceOrg,
            model=model,
            threshold=threshold
        )


        # ====================================================
        # ERROR
        # ====================================================

        if not result["success"]:

            st.error(
                "Dữ liệu giao dịch không hợp lệ."
            )


            for error in result["errors"]:

                st.error(
                    f"❌ {error}"
                )

        else:

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


            risk_level, risk_status = (
                get_risk_level(
                    probability,
                    threshold
                )
            )


            # ================================================
            # SAVE HISTORY
            # ================================================

            save_history(
                transaction_type=transaction_type,
                amount=amount,
                oldbalanceOrg=oldbalanceOrg,
                probability=probability,
                threshold=threshold,
                risk_level=risk_level,
                prediction=prediction,
                warnings=warnings
            )


            # ================================================
            # RESULT
            # ================================================

            st.divider()

            st.subheader(
                "Kết quả phân tích"
            )


            # ================================================
            # BUSINESS WARNING
            # ================================================

            if warnings:

                st.warning(
                    "Phát hiện điểm cần chú ý "
                    "về nghiệp vụ."
                )


                for warning in warnings:

                    st.write(
                        f"⚠️ {warning}"
                    )


            # ================================================
            # METRICS
            # ================================================

            col1, col2, col3 = (
                st.columns(3)
            )


            with col1:

                st.metric(
                    "Xác suất mô hình",
                    f"{probability * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Ngưỡng cảnh báo",
                    f"{threshold * 100:.0f}%"
                )


            with col3:

                st.metric(
                    "Mức độ rủi ro",
                    risk_level
                )


            # ================================================
            # RISK BAR
            # ================================================

            st.write(
                "### Mức độ rủi ro"
            )


            st.progress(
                min(
                    max(
                        float(
                            probability
                        ),
                        0.0
                    ),
                    1.0
                )
            )


            # ================================================
            # FINAL MESSAGE
            # ================================================

            if risk_status == "fraud":

                st.error(
                    "🚨 Model cảnh báo giao dịch "
                    "có nguy cơ gian lận rất cao."
                )


            elif risk_status == "high":

                st.warning(
                    "⚠️ Giao dịch có mức rủi ro cao. "
                    "Xác suất chưa vượt ngưỡng cảnh báo "
                    "nhưng nên được kiểm tra thêm."
                )


            elif risk_status == "medium":

                st.warning(
                    "⚠️ Giao dịch cần chú ý."
                )


            else:

                st.success(
                    "✅ Giao dịch có mức rủi ro thấp."
                )


            # ================================================
            # BUSINESS INFO
            # ================================================

            if warnings:

                st.info(
                    "Cảnh báo nghiệp vụ và kết quả "
                    "của model được đánh giá độc lập."
                )


            # ================================================
            # MODEL INPUT
            # ================================================

            with st.expander(
                "Xem dữ liệu được đưa vào model"
            ):

                st.dataframe(
                    transaction,
                    use_container_width=True,
                    hide_index=True
                )


            # ================================================
            # TECHNICAL INFO
            # ================================================

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


# ============================================================
# TAB 2 - DASHBOARD
# ============================================================

with tab_dashboard:

    st.subheader(
        "Dashboard giao dịch"
    )


    history_df = load_history()


    if history_df.empty:

        st.info(
            "Chưa có dữ liệu giao dịch."
        )

    else:

        # ====================================================
        # SUMMARY
        # ====================================================

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
                    "CẢNH BÁO NGUY CƠ GIAN LẬN"
                ]
            )
            .sum()
        )


        avg_probability = (
            history_df[
                "probability_percent"
            ].mean()
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "Tổng giao dịch",
                total_transactions
            )


        with col2:

            st.metric(
                "Cảnh báo",
                fraud_count
            )


        with col3:

            st.metric(
                "Rủi ro cao",
                high_risk_count
            )


        with col4:

            st.metric(
                "Xác suất trung bình",
                f"{avg_probability:.2f}%"
            )


        st.divider()


        # ====================================================
        # TYPE CHART
        # ====================================================

        st.write(
            "### Giao dịch theo loại"
        )


        type_counts = (
            history_df[
                "type"
            ]
            .value_counts()
        )


        st.bar_chart(
            type_counts
        )


        # ====================================================
        # RISK CHART
        # ====================================================

        st.write(
            "### Phân bố mức độ rủi ro"
        )


        risk_counts = (
            history_df[
                "risk_level"
            ]
            .value_counts()
        )


        st.bar_chart(
            risk_counts
        )


# ============================================================
# TAB 3 - HISTORY
# ============================================================

with tab_history:

    st.subheader(
        "Lịch sử kiểm tra giao dịch"
    )


    history_df = load_history()


    if history_df.empty:

        st.info(
            "Chưa có giao dịch nào."
        )

    else:

        col1, col2 = st.columns(2)


        # ====================================================
        # TYPE FILTER
        # ====================================================

        with col1:

            filter_type = st.selectbox(
                "Lọc theo loại giao dịch",
                [
                    "TẤT CẢ"
                ]
                +
                sorted(
                    history_df[
                        "type"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )


        # ====================================================
        # RISK FILTER
        # ====================================================

        with col2:

            filter_risk = st.selectbox(
                "Lọc theo mức độ rủi ro",
                [
                    "TẤT CẢ"
                ]
                +
                sorted(
                    history_df[
                        "risk_level"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )


        filtered_df = (
            history_df.copy()
        )


        if filter_type != "TẤT CẢ":

            filtered_df = filtered_df[
                filtered_df[
                    "type"
                ] == filter_type
            ]


        if filter_risk != "TẤT CẢ":

            filtered_df = filtered_df[
                filtered_df[
                    "risk_level"
                ] == filter_risk
            ]


        # ====================================================
        # NEWEST FIRST
        # ====================================================

        filtered_df = (
            filtered_df
            .iloc[::-1]
        )


        st.dataframe(
            filtered_df[
                [
                    "timestamp",
                    "type",
                    "amount",
                    "oldbalanceOrg",
                    "probability_percent",
                    "risk_level",
                    "warnings"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        st.caption(
            f"Hiển thị "
            f"{len(filtered_df)} giao dịch."
        )


# ============================================================
# TAB 4 - MODEL INFORMATION
# ============================================================

with tab_model:

    st.subheader(
        "Thông tin XGBoost Deployment V2"
    )


    st.markdown(
        """
        ### Feature được sử dụng

        Model triển khai cuối sử dụng 3 feature:

        - `type`
        - `amount`
        - `oldbalanceOrg`

        `step` đã được loại bỏ vì đây là thuộc tính
        thời gian đặc thù của dữ liệu mô phỏng PaySim
        và không phù hợp để tự sinh trong môi trường thực tế.
        """
    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.write(
        "### Kết quả trên tập Test"
    )


    metric_df = pd.DataFrame(
        {
            "Chỉ số": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC",
                "PR-AUC"
            ],

            "Giá trị": [
                0.999305,
                0.717341,
                0.762175,
                0.739079,
                0.998724,
                0.809026
            ]
        }
    )


    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.write(
        "### Confusion Matrix"
    )


    confusion_df = pd.DataFrame(
        {
            "Metric": [
                "True Negative",
                "False Positive",
                "False Negative",
                "True Positive"
            ],

            "Value": [
                952791,
                370,
                293,
                939
            ]
        }
    )


    st.dataframe(
        confusion_df,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "Ngưỡng cảnh báo được lựa chọn "
        "trên tập Validation bằng F1-score "
        "và có giá trị 0.99."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Financial Fraud Detection Project | "
    "XGBoost Deployment V2 | "
    "Model được huấn luyện trên dữ liệu PaySim. "
    "Kết quả là dự đoán của mô hình, "
    "không phải kết luận tuyệt đối."
)