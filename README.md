# Financial Fraud Detection Project

Hệ thống phát hiện giao dịch tài chính bất thường sử dụng Machine Learning và giao diện Streamlit.

## Model triển khai chính

Project sử dụng **XGBoost Deployment V2** làm model chính.

### Feature của V2

- `type`
- `amount`
- `oldbalanceOrg`

### Kết quả trên tập Test

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 99.9305% |
| Precision | 71.7341% |
| Recall | 76.2175% |
| F1-score | 73.9079% |
| ROC-AUC | 99.8724% |
| PR-AUC | 80.9026% |
| True Negative (TN) | 952,791 |
| False Positive (FP) | 370 |
| False Negative (FN) | 293 |
| True Positive (TP) | 939 |
| Threshold | 0.99 |

Deployment V3 có thử nghiệm thêm feature `transactions_per_hour`.

Kết quả cho thấy V3 giúp Recall và PR-AUC tăng nhẹ, tuy nhiên Precision và F1-score giảm nhẹ, đồng thời số False Positive tăng. Vì vậy **Deployment V2 được lựa chọn làm model triển khai chính**.

## Cấu trúc project

```text
Financial-Fraud-Detection_Project/
├── app/
│   ├── streamlit_app.py
│   └── services/
│       └── fraud_service.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic/
│   ├── random_forest/
│   └── xgboost/
│
├── plots/
├── reports/
├── results/
│
├── scripts/
│   ├── config/
│   ├── deployment/
│   ├── evaluation/
│   ├── experiments/
│   ├── preprocessing/
│   └── training/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Môi trường đã kiểm thử

Project đã được kiểm thử với:

- Python 3.12.6
- Streamlit 1.65.0
- pandas 3.0.6
- NumPy 2.5.3
- scikit-learn 1.9.1
- XGBoost 3.4.1
- joblib 1.6.0

Các dependency cần thiết được khai báo trong file:

```text
requirements.txt
```

## Cài đặt và chạy project

### 1. Tải project

Có thể tải project bằng một trong hai cách:

- Clone repository bằng Git.
- Chọn **Code → Download ZIP** trên GitHub và giải nén.

### 2. Mở Terminal / PowerShell tại thư mục project

Ví dụ:

```powershell
cd Financial-Fraud-Detection_Project
```

### 3. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 4. Kích hoạt môi trường ảo

Nếu PowerShell chặn việc thực thi script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Sau đó kích hoạt:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu thành công, Terminal sẽ xuất hiện:

```text
(.venv)
```

### 5. Cài đặt thư viện

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6. Chạy Streamlit

```powershell
python -m streamlit run app\streamlit_app.py
```

Ứng dụng thường chạy tại:

http://localhost:8501

## File cần thiết để chạy ứng dụng

Ứng dụng sử dụng XGBoost Deployment V2.

Hai file cần thiết là:

```text
models/xgboost/xgboost_deployment_v2_model.pkl
models/xgboost/xgboost_deployment_v2_threshold.txt
```

Nếu thiếu một trong hai file trên, ứng dụng sẽ không thể load model để dự đoán.

## Dataset

Project sử dụng bộ dữ liệu mô phỏng tài chính **PaySim**.

File raw ban đầu:

```text
PS_20174392719_1491204439457_log.csv
```

Do dataset có kích thước lớn, dữ liệu có thể không được đưa trực tiếp lên GitHub.

### Chỉ chạy ứng dụng

Nếu chỉ muốn chạy ứng dụng Streamlit với model đã huấn luyện sẵn thì **không cần dataset PaySim**.

### Train lại model

Nếu muốn thực hiện lại quá trình tiền xử lý và huấn luyện model, cần tải dataset PaySim và đặt file tại:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

Các script tiền xử lý, huấn luyện và đánh giá nằm trong:

```text
scripts/
```

## Các mô hình đã thử nghiệm

Trong quá trình thực hiện đề tài, các mô hình sau đã được thử nghiệm:

- Logistic Regression
- Random Forest
- XGBoost
- XGBoost Deployment V2
- XGBoost Deployment V3

XGBoost cho kết quả tốt nhất trong nhóm mô hình thử nghiệm ban đầu.

Deployment V2 được lựa chọn làm model triển khai chính vì có sự cân bằng tốt giữa hiệu năng và tính khả thi khi triển khai.

Deployment V3 được xây dựng để thử nghiệm thêm đặc trưng tần suất giao dịch theo khung giờ (`transactions_per_hour`), nhưng không mang lại cải thiện tổng thể đủ lớn để thay thế V2.

## Quy trình hoạt động của hệ thống

```text
Người dùng nhập thông tin giao dịch
            ↓
        Streamlit
            ↓
     fraud_service.py
            ↓
   Kiểm tra nghiệp vụ
            ↓
 XGBoost Deployment V2
            ↓
      Risk score
            ↓
   Ngưỡng cảnh báo
            ↓
     Mức độ rủi ro
```

## Business Validation

Ngoài kết quả Machine Learning, hệ thống còn kiểm tra một số điều kiện nghiệp vụ như:

- Số tiền giao dịch lớn hơn số dư hiện tại.
- Tài khoản có số dư bằng 0.
- Giao dịch sử dụng toàn bộ số dư.
- Số tiền giao dịch không hợp lệ.

Các cảnh báo nghiệp vụ được xử lý độc lập với kết quả dự đoán của model.

## Lưu ý

- PaySim là bộ dữ liệu mô phỏng, không phải dữ liệu giao dịch ngân hàng thực tế.
- Risk score của model không phải là kết luận tuyệt đối rằng một giao dịch là gian lận.
- Nếu chỉ chạy ứng dụng Streamlit thì không cần dataset gốc.

## Mục đích

Project được xây dựng phục vụ mục đích học tập và nghiên cứu về bài toán phát hiện giao dịch tài chính bất thường bằng Machine Learning.