# Báo cáo Giải pháp: Dự báo Tiến Độ Học Tập (Learning Progress Prediction)

## Tổng quan bài toán

- Mục tiêu: Dự báo số tín chỉ hoàn thành `TC_HOANTHANH` cho kỳ `HK1 2024–2025` của mỗi sinh viên.
- Ràng buộc thời gian: Train ≤ `HK1 2023–2024`, Valid = `HK2 2023–2024`, Test = `HK1 2024–2025`.
- Đặc thù dữ liệu: Chuỗi theo từng sinh viên (mỗi SV có số kỳ khác nhau), cần tạo đặc trưng chỉ dùng dữ liệu quá khứ để tránh leakage.

## Nguồn dữ liệu

- Hồ sơ tuyển sinh: [data/admission.csv](data/admission.csv)
- Lịch sử học tập: [data/academic_records.csv](data/academic_records.csv)
- Danh sách test: [data/test.csv](data/test.csv)
- Kết quả nộp: [submission.csv](submission.csv)

## Kiến trúc và mã nguồn

- Pipeline chính: [main.py](main.py)
- Phụ trợ & đặc trưng: [utils.py](utils.py)
- Hướng dẫn chạy: [README.md](README.md)
- Phụ thuộc: [requirements.txt](requirements.txt)

## Phương pháp (Pipeline)

1. Chuẩn hóa thời gian

- Parse `HOC_KY` về `(year_start, semester_num)` và tạo `semester_index` cho từng sinh viên để sắp xếp theo thời gian.

2. Hợp nhất dữ liệu & xử lý thiếu hợp lý

- Merge `admission.csv` vào `academic_records.csv` theo `MA_SO_SV`.
- Điền thiếu cho `DIEM_TRUNGTUYEN`, `DIEM_CHUAN` bằng median theo `NAM_TUYENSINH` (ổn định, tránh bias theo ngành).

3. Mã hóa biến phân loại

- One-hot cho `PTXT`, `TOHOP_XT` (giữ cả giá trị thiếu với `dummy_na=True`).

4. Tạo đặc trưng thời gian (chỉ dùng quá khứ)

- Dịch chuyển theo sinh viên: `prev_TCD` (TC_DANGKY kỳ trước), `prev_TCH` (TC_HOANTHANH kỳ trước), `prev_GPA`, `prev_GPA2`.
- Tỷ lệ hoàn thành kỳ trước: `past_last_complete_rate = prev_TCH / prev_TCD` (clip [0,1]).
- Lũy tiến (expanding quá khứ): `past_avg_gpa`, `past_avg_cpa`, `past_total_dangky`, `past_total_hoanthanh`, `past_ratio_hoanthanh`.
- Rolling quá khứ: `last_3_mean_complete_rate`.
- Biến động năng lực: `past_delta_gpa = prev_GPA - prev_GPA2`, `past_last_gpa = prev_GPA`.
- Thông tin chuỗi: `num_prev_semesters = semester_index - 1`.
- Đặc trưng tĩnh bổ sung: `diff_diem = DIEM_TRUNGTUYEN - DIEM_CHUAN`.

5. Chia tập theo thời gian

- Train: `(year_start < 2023)` hoặc `(year_start == 2023 & semester_num == 1)`.
- Valid: `(year_start == 2023 & semester_num == 2)`.

6. Huấn luyện mô hình

- Mô hình: LightGBM (ưu tiên cho dữ liệu tabular, tính năng early stopping).
- Tham số cơ sở: `n_estimators≈2500`, `learning_rate≈0.05`, `num_leaves≈64`.
- Nạp tham số đã tune nếu có: [best_params_lightgbm.json](best_params_lightgbm.json).
- Ma trận đặc trưng ép về numeric (`pd.to_numeric(errors="coerce").fillna(0)`).

7. Đánh giá (Valid)

- Chỉ số: `MSE`, `RMSE`, `R^2`, `MAPE` (phiên bản an toàn tránh chia cho 0).
- Kết quả minh họa: `RMSE ≈ 0.93`, `MSE ≈ 0.86`, `R^2 ≈ 0.98`, `MAPE ≈ 0.07`.

8. Suy luận & Hậu xử lý (Test)

- Lấy đặc trưng từ lịch sử gần nhất của từng sinh viên.
- Fallback cho SV mới (không lịch sử): dùng `comp_rate` trung bình theo `NAM_TUYENSINH` → `PTXT` → `TOHOP_XT` (ưu tiên theo thứ tự), nếu không có thì dùng trung bình toàn cục.
- Clip dự báo trong `[0, TC_DANGKY]` để đảm bảo hợp lý.
- Xuất [submission.csv](submission.csv).

## Lý do lựa chọn phương pháp

- Tree-based (LightGBM) xử lý tốt dữ liệu hỗn hợp (numeric + categorical), linh hoạt với đặc trưng tổng hợp từ chuỗi lịch sử.
- Thiết kế đặc trưng chỉ dùng quá khứ đảm bảo không leak thông tin.
- Việc dùng fallback theo cohort giúp kết quả ổn định cho SV mới.

## Tránh Leakage & Tính đúng đắn

- Mọi đặc trưng thời gian đều `shift(1)` hoặc `expanding` trên quá khứ.
- Chia tập theo thời gian nghiêm ngặt đúng yêu cầu đề.
- Không sử dụng giá trị kỳ mục tiêu khi tạo đặc trưng.

## Cách chạy

1. Cài đặt phụ thuộc và chạy pipeline:

```bash
pip install -r requirements.txt
python main.py
```

2. Kết quả: sinh file [submission.csv](submission.csv).

## Đề xuất cải tiến

- Tuning có kiểm soát: sweep `num_leaves`, `min_child_samples`, `feature_fraction`, `reg_lambda` dựa trên RMSE valid.
- CatBoost: thử giữ `PTXT`, `TOHOP_XT` dạng categorical tự nhiên.
- Explainability: thêm SHAP cho LightGBM (global & per-student) để hiểu đóng góp đặc trưng.
- Ensemble nhỏ: trung bình mô hình tree với baseline `TC_DANGKY * last_complete_rate` nếu giúp cải thiện RMSE.
- Data quality: kiểm tra consistency giữa GPA-CPA và tỷ lệ hoàn thành; thiết lập rule-based flags.

## Kết luận

- Pipeline hiện tại ổn định, tránh leakage, và đạt hiệu năng tốt trên valid theo thời gian.
- Hướng cải tiến nằm ở tuning hyperparameter, thử CatBoost/ensemble nhẹ và tăng cường khả năng giải thích bằng SHAP.
