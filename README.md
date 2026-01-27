Hướng Dẫn Chạy Nhanh (Tiếng Việt)

Mục tiêu: Chạy pipeline dự báo TC_HOÀN THÀNH và (tuỳ chọn) tune tham số cho LightGBM/CatBoost/XGBoost.

Yêu cầu

- Python 3.9 trở lên
- Đã có 3 file dữ liệu trong thư mục data/: admission.csv, academic_records.csv, test.csv

Cài đặt

1. Tạo môi trường ảo (khuyến nghị, PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Cài thư viện:

```powershell
pip install -r requirements.txt
```



Tune tham số (Optuna)

- Tối ưu RMSE cho từng mô hình; lưu tham số tốt nhất vào các file: best_params_lightgbm.json, best_params_catboost.json, best_params_xgboost.json. Tóm tắt phiên chạy lưu ở tuning_results.json.

```powershell
python tune.py
```

Sau khi tune, chạy lại `python main.py` để tự động nạp tham số tốt nhất (nếu file best*params*\*.json tồn tại).


Chạy pipeline dự báo (mặc định dùng tách theo thời gian)
ư
- Chạy huấn luyện và xuất dự báo cho 3 mô hình (LightGBM/CatBoost/XGBoost). Kết quả lưu ra các file: submission_lightgbm.csv, submission_catboost.csv, submission_xgboost.csv.

```powershell
python main.py
```