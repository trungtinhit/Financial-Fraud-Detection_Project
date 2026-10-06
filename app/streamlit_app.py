# ============================================================
# STREAMLIT APP - XGBOOST DEPLOYMENT V3
#
# Features:
# - type
# - amount
# - oldbalanceOrg
# - transactions_per_hour
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
    PROJECT_ROOT
    / "results"
    / "app"
)

HISTORY_FILE = (
    HISTORY_DIR
    / "transaction_history_v3.csv"
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
    transactions_per_hour,
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

                "transactions_per_hour":
                    transactions_per_hour,

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

    try:

        return pd.read_csv(
            HISTORY_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# 8. HEADER
# ============================================================

st.title(
    "🔍 Hệ thống phát hiện giao dịch bất thường"
)

st.caption(
    "Financial Fraud Detection - "
    "XGBoost Deployment Model V3"
)


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Thông tin hệ thống"
    )


    st.write(
        "**Model:** XGBoost Deployment V3"
    )


    st.write(
        "**Số feature:** 4"
    )


    st.write(
        f"**Ngưỡng cảnh báo:** "
        f"{threshold * 100:.0f}%"
    )


    st.divider()


    st.write(
        "**Feature sử dụng:**"
    )


    st.code(
        """
type
amount
oldbalanceOrg
transactions_per_hour
        """
    )


    st.info(
        "`transactions_per_hour` là số giao dịch "
        "đã xảy ra trước giao dịch hiện tại "
        "trong cùng khung giờ."
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
# TAB 1 - CHECK TRANSACTION
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
        # LEFT COLUMN
        # ====================================================

        with col1:

            transaction_type = st.selectbox(
                "Loại giao dịch",
                VALID_TYPES,
                help=(
                    "Loại giao dịch cần kiểm tra."
                )
            )


            amount = st.number_input(
                "Số tiền giao dịch",
                min_value=0.0,
                value=1000000.0,
                step=100000.0,
                format="%.2f",
                help=(
                    "Số tiền của giao dịch hiện tại."
                )
            )


        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with col2:

            oldbalanceOrg = st.number_input(
                "Số dư hiện tại của tài khoản",
                min_value=0.0,
                value=5000000.0,
                step=100000.0,
                format="%.2f",
                help=(
                    "Số dư tài khoản trước "
                    "khi giao dịch xảy ra."
                )
            )


            transactions_per_hour = st.number_input(
                "Số giao dịch trước đó trong cùng giờ",
                min_value=0,
                value=0,
                step=1,
                help=(
                    "Ví dụ nhập 2 nghĩa là tài khoản "
                    "đã thực hiện 2 giao dịch trước đó "
                    "trong cùng khung giờ."
                )
            )


        st.info(
            "Trong hệ thống thực tế, số giao dịch trong giờ "
            "nên được backend tự động tính từ lịch sử giao dịch. "
            "Ở bản demo, giá trị này được nhập thủ công."
        )


        submitted = st.form_submit_button(
            "🔎 Kiểm tra giao dịch",
            use_container_width=True
        )


    # ========================================================
    # PROCESS TRANSACTION
    # ========================================================

    if submitted:

        result = analyze_transaction(
            transaction_type=transaction_type,
            amount=amount,
            oldbalanceOrg=oldbalanceOrg,
            transactions_per_hour=transactions_per_hour,
            model=model,
            threshold=threshold
        )


        # ====================================================
        # INVALID DATA
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
                transactions_per_hour=transactions_per_hour,
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
            # BUSINESS WARNINGS
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
            # MAIN METRICS
            # ================================================

            col1, col2, col3 = (
                st.columns(3)
            )


            with col1:

                st.metric(
                    "Điểm rủi ro của mô hình",
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
            # STATUS MESSAGE
            # ================================================

            if risk_status == "fraud":

                st.error(
                    "🚨 Model cảnh báo giao dịch "
                    "có nguy cơ gian lận rất cao."
                )


            elif risk_status == "high":

                st.warning(
                    "⚠️ Giao dịch có mức rủi ro cao. "
                    "Điểm rủi ro chưa vượt ngưỡng "
                    "cảnh báo nhưng nên được kiểm tra thêm."
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
                    "Kết quả của mô hình và cảnh báo "
                    "nghiệp vụ được đánh giá độc lập."
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
                    f"Model score: "
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
                    "Transactions per hour: "
                    f"`{transactions_per_hour}`"
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
        # SUMMARY METRICS
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


        avg_frequency = (
            history_df[
                "transactions_per_hour"
            ].mean()
        )


        col1, col2, col3, col4, col5 = (
            st.columns(5)
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
                "Điểm rủi ro TB",
                f"{avg_probability:.2f}%"
            )


        with col5:

            st.metric(
                "Tần suất TB",
                f"{avg_frequency:.2f}"
            )


        st.divider()


        # ====================================================
        # TRANSACTION TYPE
        # ====================================================

        st.write(
            "### Số giao dịch theo loại"
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
        # RISK DISTRIBUTION
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


        # ====================================================
        # FREQUENCY DISTRIBUTION
        # ====================================================

        st.write(
            "### Tần suất giao dịch theo giờ"
        )


        frequency_counts = (
            history_df[
                "transactions_per_hour"
            ]
            .value_counts()
            .sort_index()
        )


        st.bar_chart(
            frequency_counts
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

        # ====================================================
        # FILTERS
        # ====================================================

        col1, col2 = st.columns(2)


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


        # ====================================================
        # TABLE
        # ====================================================

        st.dataframe(
            filtered_df[
                [
                    "timestamp",
                    "type",
                    "amount",
                    "oldbalanceOrg",
                    "transactions_per_hour",
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
# TAB 4 - MODEL INFO
# ============================================================

with tab_model:

    st.subheader(
        "Thông tin XGBoost Deployment V3"
    )


    st.markdown(
        """
### Các feature sử dụng

Model V3 sử dụng 4 đặc trưng:

- `type`: loại giao dịch
- `amount`: số tiền giao dịch
- `oldbalanceOrg`: số dư tài khoản trước giao dịch
- `transactions_per_hour`: số giao dịch trước đó trong cùng khung giờ

Feature `transactions_per_hour` được tạo từ `nameOrig`
và `step` trong dữ liệu PaySim bằng cách đếm số giao dịch
trước đó của cùng tài khoản trong cùng một `step`.
        """
    )


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
                0.999291,
                0.707244,
                0.768669,
                0.736678,
                0.998724,
                0.810759
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
                952769,
                392,
                285,
                947
            ]
        }
    )


    st.dataframe(
        confusion_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # V2 VS V3
    # ========================================================

    st.write(
        "### So sánh Deployment V2 và V3"
    )


    comparison_df = pd.DataFrame(
        {
            "Chỉ số": [
                "Precision",
                "Recall",
                "F1-score",
                "PR-AUC",
                "False Positive",
                "False Negative",
                "True Positive"
            ],

            "V2": [
                0.717341,
                0.762175,
                0.739079,
                0.809026,
                370,
                293,
                939
            ],

            "V3": [
                0.707244,
                0.768669,
                0.736678,
                0.810759,
                392,
                285,
                947
            ]
        }
    )


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "V3 tăng nhẹ Recall và PR-AUC, "
        "nhưng Precision và F1-score giảm nhẹ. "
        "Feature tần suất cung cấp thêm thông tin "
        "nhưng mức cải thiện trên PaySim còn hạn chế."
    )


    st.write(
        "### Ngưỡng cảnh báo"
    )


    st.write(
        f"Threshold được chọn trên tập Validation: "
        f"**{threshold * 100:.0f}%**"
    )


    st.caption(
        "Ngưỡng được lựa chọn dựa trên "
        "F1-score của tập Validation."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Financial Fraud Detection Project | "
    "XGBoost Deployment V3 | "
    "Dữ liệu PaySim | "
    "Kết quả của mô hình chỉ là tín hiệu hỗ trợ "
    "phát hiện giao dịch bất thường."
)