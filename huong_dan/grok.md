### Hướng Dẫn Giải Quyết Bài Toán Dự Báo Tiến Độ Học Tập (Learning Progress Prediction)

Tôi là một data scientist senior, và tôi sẽ hướng dẫn bạn từng bước để giải quyết bài toán này. Bài toán thuộc loại regression (dự báo số thực: TC_HOANTHANH dựa trên TC_DANGKY và lịch sử). Dữ liệu có tính chất time-series per student (lịch sử các kỳ học khác nhau về số lượng và thời gian), nên chúng ta cần xử lý cẩn thận để tổng hợp thông tin lịch sử mà không mất đi tính thời gian.

Tôi sẽ sử dụng Python (với pandas, scikit-learn, xgboost, v.v.) để minh họa. Vì môi trường có code_execution tool, bạn có thể chạy code trực tiếp, nhưng ở đây tôi mô tả chi tiết để bạn tự implement. Giả sử bạn làm trên Jupyter Notebook hoặc Google Colab.

#### **Bước 1: Hiểu Dữ Liệu và Chuẩn Bị Môi Trường**
- **Dữ liệu chính**:
  - `admission.csv`: Thông tin tĩnh (static features) cho mỗi sinh viên: MA_SO_SV (ID), NAM_TUYENSINH (năm nhập học), PTXT (phương thức xét tuyển), TOHOP_XT (tổ hợp), DIEM_TRUNGTUYEN (điểm trúng tuyển), DIEM_CHUAN (điểm chuẩn).
  - `academic_records.csv`: Lịch sử động (time-series): MA_SO_SV, HOC_KY (kỳ học, ví dụ "HK1 2020-2021"), CPA (cumulative GPA), GPA (kỳ này), TC_DANGKY (đăng ký), TC_HOANTHANH (hoàn thành).
    - Mỗi sinh viên có số kỳ khác nhau (2-8 kỳ), và kỳ không đồng đều (do năm nhập học khác).
  - `test.csv`: Chỉ có MA_SO_SV, HOC_KY="HK1 2024-2025", TC_DANGKY. Cần dự báo TC_HOANTHANH.
  - Tập train: Tất cả record đến HK1 2023-2024.
  - Tập valid: Chỉ record của HK2 2023-2024 (dùng để tune model).
- **Thách thức**:
  - Lịch sử không đồng đều: Cần aggregate (tổng hợp) lịch sử thành features cho mỗi sample.
  - Time-sensitive: Không leak data từ tương lai (ví dụ, khi dự báo valid, chỉ dùng lịch sử trước HK2 2023-2024).
  - Categorical: PTXT, TOHOP_XT cần encode.
  - Dữ liệu ngoài: Có thể thêm (ví dụ, điểm chuẩn các năm từ web), nhưng ở đây tôi tập trung vào data có sẵn.

- **Setup môi trường** (chạy trong notebook):
  ```python
  import pandas as pd
  import numpy as np
  from sklearn.preprocessing import OneHotEncoder, StandardScaler
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
  from sklearn.linear_model import LinearRegression
  from xgboost import XGBRegressor
  import shap  # For explainability
  import matplotlib.pyplot as plt
  ```

- **Đọc dữ liệu**:
  ```python
  admission = pd.read_csv('admission.csv')
  records = pd.read_csv('academic_records.csv')
  test = pd.read_csv('test.csv')
  # Kiểm tra shape: admission (~unique MA_SO_SV), records (multiple rows per SV)
  print(admission.shape, records.shape, test.shape)
  ```

#### **Bước 2: Xử Lý Dữ Liệu (Data Processing)**
Mục tiêu: Tạo một dataframe flat (một row per sample) với features từ admission + aggregate lịch sử + TC_DANGKY.

1. **Xử lý HOC_KY (thời gian)**:
   - Parse HOC_KY thành year và semester (HK1=1, HK2=2).
   - Tạo cột 'semester_order': Sắp xếp kỳ theo thứ tự thời gian cho mỗi sinh viên (ví dụ, kỳ đầu=1, kỳ sau=2).
   ```python
   def parse_hoc_ky(hk):
       if pd.isna(hk): return np.nan, np.nan
       parts = hk.split()
       semester = 1 if parts[0] == 'HK1' else 2
       year_start = int(parts[1].split('-')[0])
       return year_start, semester
   
   records[['year', 'semester']] = records['HOC_KY'].apply(parse_hoc_ky).apply(pd.Series)
   records = records.sort_values(['MA_SO_SV', 'year', 'semester'])  # Sort by time per student
   records['semester_order'] = records.groupby('MA_SO_SV').cumcount() + 1  # Order 1,2,3,... per SV
   ```

2. **Merge admission và records**:
   - Left join admission vào records trên MA_SO_SV.
   ```python
   df = pd.merge(records, admission, on='MA_SO_SV', how='left')
   ```

3. **Feature Engineering**:
   - **Static features (từ admission)**:
     - 'diff_diem': DIEM_TRUNGTUYEN - DIEM_CHUAN (mức vượt chuẩn).
     - Encode categorical: One-hot cho PTXT, TOHOP_XT.
     ```python
     df['diff_diem'] = df['DIEM_TRUNGTUYEN'] - df['DIEM_CHUAN']
     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
     cat_features = encoder.fit_transform(df[['PTXT', 'TOHOP_XT']])
     cat_df = pd.DataFrame(cat_features, columns=encoder.get_feature_names_out())
     df = pd.concat([df.reset_index(drop=True), cat_df], axis=1)
     df.drop(['PTXT', 'TOHOP_XT'], axis=1, inplace=True)  # Drop original
     ```
   - **Dynamic features (từ lịch sử)**: Tổng hợp lịch sử TRƯỚC kỳ dự báo cho mỗi sample.
     - Vì dự báo cho một kỳ cụ thể, cần tạo features như: avg_GPA trước, total_TC_hoanthanh trước, ratio_hoanthanh (TC_HOANTHANH / TC_DANGKY avg trước), trend_GPA (slope của linear reg trên GPA theo semester_order).
     - Sử dụng groupby + shift để tránh data leak (chỉ dùng past data).
     ```python
     # Tính aggregate past per row
     df['past_avg_gpa'] = df.groupby('MA_SO_SV')['GPA'].shift(1).rolling(window=100).mean()  # Mean of all past GPA
     df['past_avg_cpa'] = df.groupby('MA_SO_SV')['CPA'].shift(1).rolling(window=100).mean()
     df['past_total_dangky'] = df.groupby('MA_SO_SV')['TC_DANGKY'].shift(1).cumsum()
     df['past_total_hoanthanh'] = df.groupby('MA_SO_SV')['TC_HOANTHANH'].shift(1).cumsum()
     df['past_ratio_hoanthanh'] = df['past_total_hoanthanh'] / df['past_total_dangky']
     df.fillna(0, inplace=True)  # For first semester, past=0

     # Trend: Slope of GPA over time (simple linear reg per student)
     def calc_slope(group):
         if len(group) < 2: return 0
         model = LinearRegression().fit(np.array(group['semester_order']).reshape(-1,1), group['GPA'])
         return model.coef_[0]
     slopes = df.groupby('MA_SO_SV').apply(calc_slope).reset_index(name='gpa_trend')
     df = pd.merge(df, slopes, on='MA_SO_SV')
     ```
   - Thêm features khác: Số kỳ đã học (semester_order - 1), NAM_TUYENSINH (năm nhập học).

4. **Tách Train/Valid/Test**:
   - Train: Records với year <= 2023 và semester=1 (HK1 2023-2024) hoặc trước.
   - Valid: Records với HOC_KY == 'HK2 2023-2024'.
   - Test: Merge test với admission và aggregate toàn bộ lịch sử (vì test là tương lai).
   ```python
   train_df = df[(df['year'] < 2023) | ((df['year'] == 2023) & (df['semester'] == 1))]
   valid_df = df[df['HOC_KY'] == 'HK2 2023-2024']
   
   # For test: Aggregate full history per SV in test
   test_hist = df.groupby('MA_SO_SV').agg({
       'past_avg_gpa': 'last', 'past_avg_cpa': 'last', 'past_total_hoanthanh': 'last', 
       'gpa_trend': 'first', 'diff_diem': 'first', 'NAM_TUYENSINH': 'first'
       # Add one-hot cols similarly
   }).reset_index()
   test_df = pd.merge(test, test_hist, on='MA_SO_SV', how='left')
   test_df['semester_order'] = df.groupby('MA_SO_SV')['semester_order'].max() + 1  # Next semester
   ```

5. **Xử lý missing/outliers**:
   - Fillna: 0 cho past features nếu không có lịch sử.
   - Scale numerical: StandardScaler cho ['GPA', 'CPA', 'TC_DANGKY', 'diff_diem', v.v.].
   - Drop columns không cần: HOC_KY, year, semester (sau khi dùng).

#### **Bước 3: Chọn và Huấn Luyện Mô Hình**
- **Features (X)**: past_avg_gpa, past_avg_cpa, past_ratio_hoanthanh, gpa_trend, TC_DANGKY, diff_diem, NAM_TUYENSINH, one-hot PTXT/TOHOP.
- **Target (y)**: TC_HOANTHANH.
- **Mô hình cơ bản**: Linear Regression (dễ explain).
- **Mô hình nâng cao**: XGBoost (handle non-linear, tốt cho tabular data).
  ```python
  features = ['past_avg_gpa', 'past_avg_cpa', 'past_ratio_hoanthanh', 'gpa_trend', 'TC_DANGKY', 'diff_diem', 'NAM_TUYENSINH'] + list(cat_df.columns)
  X_train = train_df[features]
  y_train = train_df['TC_HOANTHANH']
  X_valid = valid_df[features]
  y_valid = valid_df['TC_HOANTHANH']
  X_test = test_df[features]

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_valid = scaler.transform(X_valid)
  X_test = scaler.transform(X_test)

  # Train XGBoost
  model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
  model.fit(X_train, y_train)

  # Predict valid
  y_pred_valid = model.predict(X_valid)
  print('R2:', r2_score(y_valid, y_pred_valid))
  print('RMSE:', np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
  print('MSE:', mean_squared_error(y_valid, y_pred_valid))
  print('MAPE:', mean_absolute_percentage_error(y_valid, y_pred_valid))
  ```
- **Tune hyperparams**: Sử dụng GridSearchCV trên valid set.
- **Sequence model (nâng cao)**: Nếu muốn exploit time-series đầy đủ, dùng LSTM (pad sequences lịch sử thành fixed length).
  - Input: [lịch sử GPA, CPA, TC_DANGKY/HOANTHANH của max 8 kỳ] + static features.
  - Nhưng XGBoost thường đủ tốt cho bài này.

#### **Bước 4: Explainable AI**
- Sử dụng SHAP để xem feature importance.
  ```python
  explainer = shap.Explainer(model)
  shap_values = explainer(X_valid)
  shap.summary_plot(shap_values, X_valid, feature_names=features)  # Plot
  ```
- Hoặc LIME cho instance-level.

#### **Bước 5: Dự Báo Test và Submit**
- Predict trên X_test.
  ```python
  y_pred_test = model.predict(X_test)
  submission = pd.DataFrame({'MA_SO_SV': test['MA_SO_SV'], 'PRED_TC_HOANTHANH': y_pred_test})
  submission.to_csv('ten_doi_thi.csv', index=False)
  ```

#### **Bước 6: Cải Tiến và Lưu Ý**
- **Thêm data ngoài**: Search web cho điểm chuẩn các năm (browse_page tool với url như "diem chuan dai hoc 2020-2024"), thêm feature như "avg_diem_chuan_year".
- **Handle imbalance**: Nếu TC_HOANTHANH skewed, dùng log transform.
- **Reproducible**: Seed random, save model (joblib).
- **Dashboard**: Sử dụng Streamlit hoặc Tableau cho valid set (plot dự báo vs actual).
- **Thời gian**: 20p trình bày: Focus logic, metrics, SHAP.
- **Lỗi thường gặp**: Data leak (dùng future data), quên encode categorical.

Nếu cần code đầy đủ hoặc chạy cụ thể, cung cấp thêm chi tiết!