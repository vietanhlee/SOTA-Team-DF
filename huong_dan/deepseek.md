Tôi sẽ hướng dẫn bạn giải quyết bài toán **Learning Progress Prediction** từ góc độ Data Scientist, với các bước chi tiết từ xử lý dữ liệu đến huấn luyện mô hình.

---

## **1. PHÂN TÍCH BÀI TOÁN & ĐẶC THÙ DỮ LIỆU**

### **Bài toán**: 
- **Regression** (dự báo số tín chỉ hoàn thành - `TC_HOANTHANH`)
- **Time-aware prediction**: Mỗi sinh viên có lịch sử học tập khác nhau, số kỳ khác nhau, năm học khác nhau.
- **Đầu vào**: 
  - Thông tin tuyển sinh cố định (admission.csv)
  - Lịch sử học tập theo kỳ (academic_records.csv)
  - Số tín chỉ đăng ký kỳ hiện tại (`TC_DANGKY`)

---

## **2. KIẾN TRÚC GIẢI QUYẾT TỔNG QUAN**

```
Data Collection → Feature Engineering → Model Training → Prediction → Interpretation
```

---

## **3. CHI TIẾT CÁC BƯỚC XỬ LÝ**

### **3.1. Phân tích & Tiền xử lý dữ liệu**

#### **a. Đọc và khám phá dữ liệu**
```python
import pandas as pd
import numpy as np

# Đọc dữ liệu
admission = pd.read_csv('admission.csv')
academic = pd.read_csv('academic_records.csv')

# Kiểm tra missing values, outliers
print(admission.info())
print(academic.info())
```

#### **b. Xử lý dữ liệu thời gian**
```python
# Chuyển HOC_KY thành dạng số để so sánh
def convert_hoc_ky(hoc_ky_str):
    # Ví dụ: "2020-2021-1" → (2020, 1)
    year, semester = hoc_ky_str.split('-')[0], hoc_ky_str.split('-')[-1]
    return int(year), int(semester)

academic['NAM_HOC'], academic['HOC_KY_SO'] = zip(*academic['HOC_KY'].apply(convert_hoc_ky))
academic['TIME_ORDER'] = academic['NAM_HOC'] * 2 + academic['HOC_KY_SO'] - 1
```

#### **c. Xử lý missing values**
- Điểm số (CPA, GPA): fill bằng trung bình theo ngành/năm
- TC_HOANTHANH: không có missing (vì là nhãn trong train)
- Admission: kiểm tra và fill hợp lý

---

### **3.2. Feature Engineering - BƯỚC QUAN TRỌNG NHẤT**

#### **a. Features từ admission.csv**
```python
# 1. Độ chênh điểm so với chuẩn
admission['CHENH_LECH_DIEM'] = admission['DIEM_TRUNGTUYEN'] - admission['DIEM_CHUAN']

# 2. Mã hóa categorical (PTXT, TOHOP_XT) bằng Target Encoding hoặc One-Hot
from sklearn.preprocessing import LabelEncoder

le_ptxt = LabelEncoder()
admission['PTXT_ENCODED'] = le_ptxt.fit_transform(admission['PTXT'])
```

#### **b. Features từ lịch sử học tập (Time-aware Aggregation)**
Với **mỗi sinh viên ở mỗi kỳ học**, ta tính features từ các kỳ **TRƯỚC ĐÓ**:

```python
def create_historical_features(student_id, target_time_order):
    # Lấy tất cả records của sinh viên TRƯỚC kỳ đang xét
    hist = academic[(academic['MA_SO_SV'] == student_id) & 
                    (academic['TIME_ORDER'] < target_time_order)]
    
    features = {}
    
    if len(hist) > 0:
        # 1. Performance statistics
        features['CPA_MEAN'] = hist['CPA'].mean()
        features['CPA_STD'] = hist['CPA'].std()
        features['GPA_MEAN'] = hist['GPA'].mean()
        features['GPA_TREND'] = np.polyfit(range(len(hist)), hist['GPA'], 1)[0] if len(hist) > 1 else 0
        
        # 2. Credit completion history
        hist['COMPLETION_RATE'] = hist['TC_HOANTHANH'] / hist['TC_DANGKY']
        features['COMPLETION_MEAN'] = hist['COMPLETION_RATE'].mean()
        features['COMPLETION_STD'] = hist['COMPLETION_RATE'].std()
        
        # 3. Accumulated credits
        features['TOTAL_TC_DANGKY'] = hist['TC_DANGKY'].sum()
        features['TOTAL_TC_HOANTHANH'] = hist['TC_HOANTHANH'].sum()
        features['OVERALL_COMPLETION_RATE'] = features['TOTAL_TC_HOANTHANH'] / features['TOTAL_TC_DANGKY']
        
        # 4. Recent performance (last semester)
        last_sem = hist.iloc[-1]
        features['LAST_CPA'] = last_sem['CPA']
        features['LAST_COMPLETION'] = last_sem['TC_HOANTHANH'] / last_sem['TC_DANGKY']
        
        # 5. Semester count
        features['NUM_PREV_SEMESTERS'] = len(hist)
    else:
        # For new students (first semester)
        features = {key: 0 for key in ['CPA_MEAN', 'CPA_STD', 'GPA_MEAN', 'GPA_TREND',
                                       'COMPLETION_MEAN', 'COMPLETION_STD',
                                       'TOTAL_TC_DANGKY', 'TOTAL_TC_HOANTHANH',
                                       'OVERALL_COMPLETION_RATE', 'LAST_CPA',
                                       'LAST_COMPLETION', 'NUM_PREV_SEMESTERS']}
        features['IS_NEW_STUDENT'] = 1
    
    return features
```

#### **c. Features kỳ hiện tại**
- `TC_DANGKY` (số tín chỉ đăng ký kỳ này)
- `HOC_KY_SO` (học kỳ 1 hay 2)
- `NAM_HOC` (năm học)

---

### **3.3. Tạo dataset huấn luyện**

```python
def prepare_dataset(admission_df, academic_df, mode='train'):
    """
    mode: 'train', 'valid', hoặc 'test'
    """
    dataset = []
    
    for idx, row in academic_df.iterrows():
        student_id = row['MA_SO_SV']
        time_order = row['TIME_ORDER']
        
        # 1. Lấy thông tin admission
        admission_info = admission_df[admission_df['MA_SO_SV'] == student_id].iloc[0]
        
        # 2. Tạo historical features
        hist_features = create_historical_features(student_id, time_order)
        
        # 3. Kết hợp tất cả features
        features = {
            'MA_SO_SV': student_id,
            'TIME_ORDER': time_order,
            # Admission features
            'NAM_TUYENSINH': admission_info['NAM_TUYENSINH'],
            'PTXT_ENCODED': admission_info['PTXT_ENCODED'],
            'CHENH_LECH_DIEM': admission_info['CHENH_LECH_DIEM'],
            # Current semester
            'TC_DANGKY': row['TC_DANGKY'],
            'HOC_KY_SO': row['HOC_KY_SO'],
            'NAM_HOC': row['NAM_HOC'],
            # Historical features
            **hist_features
        }
        
        # 4. Thêm target (chỉ với train/valid)
        if mode != 'test':
            features['TC_HOANTHANH'] = row['TC_HOANTHANH']
        
        dataset.append(features)
    
    return pd.DataFrame(dataset)
```

---

### **3.4. Chia tập dữ liệu theo đề bài**

```python
# Academic records gốc đã có sẵn TIME_ORDER
# Train: TIME_ORDER < [ngưỡng HK2 2023-2024]
# Valid: TIME_ORDER = [ngưỡng HK2 2023-2024]
# Test: Dựa trên file test.csv, cần thêm thông tin historical từ train+valid

train_academic = academic[academic['TIME_ORDER'] < valid_threshold]
valid_academic = academic[academic['TIME_ORDER'] == valid_threshold]

train_df = prepare_dataset(admission, train_academic, mode='train')
valid_df = prepare_dataset(admission, valid_academic, mode='valid')
```

---

### **3.5. Xử lý dữ liệu Test**
```python
# Đọc test.csv (chỉ có MA_SO_SV và TC_DANGKY)
test_info = pd.read_csv('test.csv')

# Thêm thông tin thời gian cho kỳ test
test_info['HOC_KY_SO'] = 1  # HK1
test_info['NAM_HOC'] = 2024
test_info['TIME_ORDER'] = 2024 * 2 + 1 - 1  # Tính toán

# Tạo dataset test với historical features từ TOÀN BỘ train+valid
full_academic = pd.concat([train_academic, valid_academic])
test_df = prepare_dataset(admission, test_info, mode='test')
```

---

## **4. MÔ HÌNH HUẤN LUYỆN**

### **4.1. Các mô hình đề xuất**
1. **Gradient Boosting** (ưu tiên vì xử lý tốt mixed data type):
   - XGBoost / LightGBM / CatBoost
   - Có sẵn feature importance

2. **Random Forest**:
   - Robust với outliers
   - Dễ giải thích

3. **Neural Networks** (nếu dữ liệu đủ lớn):
   - MLP với embedding cho categorical

4. **Ensemble**: Kết hợp nhiều mô hình

### **4.2. Triển khai với LightGBM**
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Chia features và target
X_train = train_df.drop(['MA_SO_SV', 'TC_HOANTHANH'], axis=1)
y_train = train_df['TC_HOANTHANH']
X_valid = valid_df.drop(['MA_SO_SV', 'TC_HOANTHANH'], axis=1)
y_valid = valid_df['TC_HOANTHANH']

# Huấn luyện
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    importance_type='gain'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=50,
    verbose=50
)

# Dự báo
y_pred = model.predict(X_valid)

# Đánh giá
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)
mape = mean_absolute_percentage_error(y_valid, y_pred)
```

### **4.3. Post-processing**
```python
# Đảm bảo dự báo không vượt quá TC_DANGKY
test_df['PRED_TC_HOANTHANH'] = model.predict(test_df.drop('MA_SO_SV', axis=1))
test_df['PRED_TC_HOANTHANH'] = test_df.apply(
    lambda x: min(x['PRED_TC_HOANTHANH'], x['TC_DANGKY']), axis=1
)
test_df['PRED_TC_HOANTHANH'] = test_df['PRED_TC_HOANTHANH'].clip(lower=0)
```

---

## **5. PHÂN NHÓM SINH VIÊN RỦI RO**

```python
# Phân loại dựa trên tỷ lệ hoàn thành dự báo
test_df['PRED_COMPLETION_RATE'] = test_df['PRED_TC_HOANTHANH'] / test_df['TC_DANGKY']

# Nhóm rủi ro cao: dự báo hoàn thành < 50%
high_risk = test_df[test_df['PRED_COMPLETION_RATE'] < 0.5]
medium_risk = test_df[(test_df['PRED_COMPLETION_RATE'] >= 0.5) & 
                      (test_df['PRED_COMPLETION_RATE'] < 0.7)]
low_risk = test_df[test_df['PRED_COMPLETION_RATE'] >= 0.7]
```

---

## **6. GIẢI THÍCH MÔ HÌNH (Explainable AI)**

```python
import shap

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualization
shap.summary_plot(shap_values, X_train)
```

---

## **7. ĐỀ XUẤT CHÍNH SÁCH HỖ TRỢ**

Dựa trên feature importance và SHAP values:
1. **Sinh viên mới** (không có lịch sử): 
   - Chương trình định hướng
   - Hỗ trợ học tập từ đầu

2. **Sinh viên có `COMPLETION_STD` cao** (dao động tỷ lệ hoàn thành):
   - Theo dõi sát sao
   - Hỗ trợ tâm lý

3. **Sinh viên đăng ký quá nhiều tín chỉ**:
   - Tư vấn điều chỉnh học phần

---

## **8. TỐI ƯU HÓA & CẢI THIỆN**

### **a. Cross-validation Time-series**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X_train):
    # Training với validation theo thời gian
    pass
```

### **b. Hyperparameter Tuning**
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    # ... train và evaluate
```

### **c. Sử dụng dữ liệu external**
- Thu thập điểm chuẩn các trường khác
- Dữ liệu kinh tế xã hội theo năm
- Weather data (nếu ảnh hưởng học tập)

---

## **9. OUTPUT CUỐI CÙNG**

```python
# File submission
submission = test_df[['MA_SO_SV', 'PRED_TC_HOANTHANH']]
submission.to_csv('ten_doi_thi.csv', index=False)

# Báo cáo nhóm sinh viên rủi ro
risk_report = high_risk[['MA_SO_SV', 'PRED_COMPLETION_RATE']].sort_values('PRED_COMPLETION_RATE')
```

---

## **TÓM TẮT QUY TRÌNH**

1. **Data Understanding**: Phân tích đặc thù time-series
2. **Feature Engineering**: Tạo features từ lịch sử (quan trọng nhất)
3. **Model Selection**: Gradient Boosting với explainability
4. **Validation**: Time-series split, đánh giá đa chỉ số
5. **Interpretation**: SHAP, feature importance
6. **Application**: Phân nhóm rủi ro, đề xuất chính sách

## **LƯU Ý QUAN TRỌNG**
- **Không leakage**: Chỉ dùng dữ liệu TRƯỚC kỳ đang dự báo
- **Xử lý sinh viên mới**: Có strategy riêng
- **Reproducibility**: Fix random seed, lưu pipeline
- **Documentation**: Ghi rõ mọi quyết định xử lý

Bạn cần thực hiện từng bước, đánh giá sau mỗi bước để điều chỉnh. Ưu tiên **đơn giản → phức tạp**, bắt đầu với các features cơ bản trước.