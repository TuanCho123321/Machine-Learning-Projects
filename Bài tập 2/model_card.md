# Model Card – Bank Marketing Classification

## 1. Tổng quan

- Bài toán: Dự đoán khách hàng có đăng ký tiền gửi kỳ hạn hay không.
- Dataset: UCI Bank Marketing Dataset.
- Số dòng: 45,211.
- Số feature gốc: 16.
- Target: `y`.
- Train/Test: 80/20, stratified.
- Model cuối cùng: Random Forest.
- Model version: `bank-marketing-rf-v1.0`.

## 2. Feature Engineering

Đã tạo 5 feature mới:

- `age_group`
- `previously_contacted`
- `balance_sign`
- `campaign_intensity`
- `previous_contact_group`

Các feature được tạo trước khi chia train/test nhưng không sử dụng target để tạo feature.

## 3. Preprocessing

### Numerical features

- SimpleImputer với median.
- StandardScaler.

### Categorical features

- SimpleImputer với most_frequent.
- OneHotEncoder với `handle_unknown='ignore'`.

Preprocessing được đặt trong Pipeline nên scaler và encoder chỉ được fit trên dữ liệu training.

## 4. Xử lý mất cân bằng

Đã thử hai phương pháp:

1. Class Weight với `class_weight='balanced'`.
2. SMOTE.

SMOTE được đặt bên trong pipeline để chỉ resample dữ liệu training.

## 5. Các model đã thử

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Ngoài ra đã thử các model với class weight và SMOTE.

## 6. Model cuối cùng

- Model: Random Forest
- Phương pháp imbalance: Class Weight
- Threshold: 0.20
- Model version: `bank-marketing-rf-v1.0`

Threshold được lựa chọn trên validation data dựa trên chi phí nghiệp vụ.

## 7. Business Cost

Giả sử:

- False Negative (FN): chi phí 4.
- False Positive (FP): chi phí 1.

Công thức:

`Total Cost = 4 × FN + FP`

Threshold cuối cùng:

`0.20`

## 8. Kết quả trên Test Set

| Metric | Result |
|---|---:|
| Precision | 0.464978 |
| Recall | 0.815690 |
| F1-score | 0.592313 |
| ROC-AUC | 0.927890 |
| FP | 993 |
| FN | 195 |
| Total Cost | 1773 |
| Threshold | 0.20 |

Tính chi phí:

`4 × 195 + 993 = 1773`

## 9. Feature Importance

Top các feature quan trọng:

| Feature | Importance |
|---|---:|
| duration | 0.324051 |
| month | 0.100076 |
| balance | 0.068178 |
| day | 0.060702 |
| age | 0.059097 |
| job | 0.055507 |
| contact | 0.044139 |
| poutcome | 0.041594 |
| housing | 0.033924 |
| age_group | 0.033217 |

Feature importance thể hiện mức độ đóng góp tương đối của feature trong Random Forest, không thể hiện trực tiếp chiều tác động tăng hoặc giảm xác suất.

## 10. Ví dụ dự đoán

### Khách hàng có xác suất cao

- Age: 25
- Job: student
- Marital: single
- Education: secondary
- Balance: 469
- Housing: no
- Loan: no
- Contact: cellular
- Duration: 524
- Campaign: 1
- Pdays: 187
- Previous: 2
- Poutcome: success

Kết quả:

- Probability: 0.935
- Prediction: 1
- Opportunity level: High
- Actual: 0

Đây là một False Positive.

### Khách hàng có xác suất thấp

- Age: 44
- Job: technician
- Marital: married
- Education: secondary
- Balance: 378
- Housing: yes
- Loan: no
- Contact: unknown
- Duration: 203
- Campaign: 2
- Pdays: -1
- Previous: 0
- Poutcome: unknown

Kết quả:

- Probability: 0.0
- Prediction: 0
- Opportunity level: Low
- Actual: 0

Đây là một True Negative.

## 11. Fairness Audit

Audit được thực hiện theo nhóm tuổi và nhóm `job`.

Các chỉ số:

- Prediction Positive Rate
- Recall / TPR
- False Positive Rate

### Age Group

| Group | Positive Rate | Recall / TPR | FPR | Count |
|---|---:|---:|---:|---:|
| <=25 | 0.396887 | 0.877193 | 0.260000 | 257 |
| 26-35 | 0.216013 | 0.792553 | 0.135246 | 3060 |
| 36-45 | 0.166425 | 0.821138 | 0.102309 | 2758 |
| 46-55 | 0.165901 | 0.796954 | 0.095346 | 1959 |
| 56-65 | 0.222607 | 0.803279 | 0.127517 | 867 |
| >65 | 0.816901 | 0.966667 | 0.707317 | 142 |

### Job

| Job | Positive Rate | Recall / TPR | FPR | Count |
|---|---:|---:|---:|---:|
| admin. | 0.194175 | 0.786885 | 0.114537 | 1030 |
| blue-collar | 0.129338 | 0.788321 | 0.078187 | 1902 |
| entrepreneur | 0.149502 | 0.727273 | 0.103943 | 301 |
| housemaid | 0.158590 | 0.833333 | 0.078818 | 227 |
| management | 0.237288 | 0.803846 | 0.149970 | 1947 |
| retired | 0.413717 | 0.910000 | 0.272727 | 452 |
| self-employed | 0.208333 | 0.822222 | 0.113402 | 336 |
| services | 0.150124 | 0.774648 | 0.089796 | 806 |
| student | 0.511111 | 0.920000 | 0.353846 | 180 |
| technician | 0.202747 | 0.813559 | 0.122781 | 1529 |
| unemployed | 0.250896 | 0.795455 | 0.148936 | 279 |
| unknown | 0.314815 | 1.000000 | 0.229167 | 54 |

Có sự khác biệt về các chỉ số giữa các nhóm. Những nhóm có số lượng mẫu nhỏ cần được diễn giải thận trọng.

## 12. API

Endpoint:

`POST /predict`

API nhận 16 feature gốc dưới dạng JSON.

API tự tạo 5 feature engineering giống notebook trước khi đưa dữ liệu vào pipeline.

Response gồm:

- `prediction`
- `probability`
- `threshold`
- `model_version`
- `opportunity_level`

Model version:

`bank-marketing-rf-v1.0`

## 13. Limitations

- Dataset có class imbalance.
- Threshold 0.20 phụ thuộc vào giả định chi phí FN = 4 và FP = 1.
- Feature importance không cho biết trực tiếp chiều tác động.
- Một số nhóm fairness có số lượng mẫu nhỏ.
- Khi triển khai thực tế cần theo dõi data drift và hiệu năng mô hình theo thời gian.
