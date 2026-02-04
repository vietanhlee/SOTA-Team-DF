Hướng Dẫn Chạy Nhanh (Tiếng Việt)

Project này **chỉ dùng XGBoost** để dự đoán số tín chỉ hoàn thành.

Luồng chuẩn:

1. Chạy notebook `training.ipynb` để tạo/kiểm tra `best_params_xgboost.json` và sinh thử `submissionn.csv`.
2. Chạy app Streamlit `streamlit_app.py` để upload CSV → predict → download CSV (kèm ghi chú SHAP).

## Yêu cầu

- Python 3.9+ (khuyến nghị 3.10/3.11)
- Có sẵn 3 file dữ liệu trong `data/`:
  - `data/academic_records.csv`
  - `data/admission.csv`
  - `data/test.csv`

## Cài đặt

PowerShell (Windows):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Bước 1 — Chạy notebook training

Mở và chạy notebook:

- File: `training.ipynb`
- Notebook sẽ:
  - Dùng `pipeline.build_main_df()` để tạo feature engineering (đồng bộ với app)
  - (Tuỳ chọn) tune XGBoost bằng Optuna và lưu `best_params_xgboost.json`
  - Train final model và xuất `submissionn.csv` từ `data/test.csv`

Gợi ý biến môi trường (tuỳ chọn):

- `N_TRIALS`: số lần thử Optuna (mặc định 30)
- `USE_CUDA=1`: bật GPU nếu XGBoost hỗ trợ CUDA
- `VALID_HK`: học kỳ dùng làm validation (mặc định `HK2 2023-2024`)

Ví dụ chạy JupyterLab:

```powershell
jupyter lab
```

Sau khi chạy xong, bạn sẽ có:

- `best_params_xgboost.json` (nếu bạn chạy cell Optuna)
- `submissionn.csv`

## Bước 2 — Chạy Streamlit app

App nhận file CSV giống format `data/test.csv`.

- Input tối thiểu cần 2 cột:
  - `MA_SO_SV`
  - `TC_DANGKY`
- `HOC_KY` là tuỳ chọn

Output khi download CSV:

- `PRED_PASS_RATE`: tỷ lệ hoàn thành dự đoán
- `PRED_TC_HOANTHANH`: số tín chỉ hoàn thành dự đoán
- `GHI_CHU_SHAP`: ghi chú giải thích top feature ảnh hưởng theo SHAP

Chạy app:

```powershell
streamlit run streamlit_app.py
```

Lưu ý: App sẽ tự load `data/*.csv`, train model theo `best_params_xgboost.json` (nếu có) và tính SHAP để tạo cột ghi chú.
