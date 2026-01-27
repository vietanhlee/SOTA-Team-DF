# Kế hoạch toàn diện (từ xử lý dữ liệu → huấn luyện → đánh giá → triển khai)

Dưới đây là lộ trình chi tiết — vừa kỹ thuật vừa thực tế — để giải bài **Dự báo số tín chỉ hoàn thành (TC_HOANTHANH)** cho mỗi sinh viên. Mình sẽ nói rõ cách xử lý dữ liệu chuỗi thời gian nhiều chiều (mỗi sinh viên có số kỳ khác nhau), các lựa chọn mô hình (từ baseline đến model tuần tự), chiến lược đánh giá theo thời gian, và cách triển khai/giải thích kết quả. (Các quy định tập train/valid/test và chỉ số đánh giá lấy theo đề bài). 

---

# 1) Tóm tắt bài toán & dữ liệu (nhanh)

* Mục tiêu: với mỗi sinh viên và học kỳ sắp tới (test = HK1 2024–2025), dự báo `TC_HOANTHANH` (số tín chỉ hoàn thành cuối kỳ) dựa trên:

  * thông tin tuyển sinh (admission.csv),
  * lịch sử học tập theo kỳ (academic_records.csv),
  * và `TC_DANGKY` của kỳ cần dự báo.
* Tập train/valid/test được quy định theo thời gian (không random split). 

---

# 2) Tổng quan chiến lược (một nhìn)

1. **Khám phá & làm sạch (EDA)** — hiểu phân phối tín chỉ, thiếu dữ liệu, outlier theo cá nhân và theo kỳ.
2. **Chuẩn hóa timeline** — chuyển `HOC_KY` thành chỉ số thời gian (semester index) cố định/tuần tự để sắp xếp lịch sử của mỗi sinh viên.
3. **Feature engineering** — hai luồng:

   * *sequence-aware features* (dùng cho RNN/Transformer): giữ chuỗi kỳ cho mỗi sinh viên (GPA, CPA, TC_DANGKY, TC_HOANTHANH, delta, thời gian giữa các kỳ,...).
   * *aggregated snapshot features* (dùng cho LightGBM/Linear): các thống kê lịch sử (mean, std, last, trend, cum_sum, ratio hoàn thành, slope), cùng các feature tĩnh (admission).
4. **Baseline → mạnh hơn**:

   * baseline đơn giản: `pred = TC_DANGKY * mean_completion_rate_last_k`
   * tree-based (LightGBM/XGBoost) trên features tổng hợp — thường rất mạnh.
   * sequence models: LSTM/GRU hoặc Transformer (Temporal Fusion Transformer) dùng khi chuỗi có thông tin thời điểm quan trọng.
5. **Đánh giá**: dùng validation theo thời gian (HK2 2023–2024 theo đề) và metric bắt buộc: `MSE, RMSE, R^2, MAPE`. 
6. **Giải thích & ứng dụng**: SHAP cho tree models; attention/feature importance cho sequence models; thresholding để phát hiện học sinh “nguy cơ” (ví dụ: pred < 0.8 * TC_DANGKY).

---

# 3) Bước-1: EDA & chuẩn bị dữ liệu

Các việc cần làm:

* Load 3 file: `admission.csv`, `academic_records.csv`, `test.csv` (test chỉ có MA_SO_SV + TC_DANGKY).
* Kiểm tra:

  * Số sinh viên, phân bố số kỳ / sinh viên (`count` per MA_SO_SV).
  * Phân bố `TC_DANGKY`, `TC_HOANTHANH` (min, max, mean). Tìm outlier (ví dụ TC_HOANTHANH > TC_DANGKY — xác minh).
  * Các giá trị thiếu ở `GPA/CPA/DIEM_TRUNGTUYEN` → xử lý.
* Chuẩn hóa `HOC_KY`:

  * Nếu `HOC_KY` là chuỗi (Ví dụ: "HK1_2020-2021"), parse thành thứ tự thời gian (semester_idx) để sắp xếp chính xác. Nếu chỉ là số, map theo năm học.
  * Tạo `semester_index` tăng dần (0,1,2,...) để dễ pack seq.
* Merge admission vào academic_records theo `MA_SO_SV`.

---

# 4) Xử lý dữ liệu thiếu và loạn

* Nếu `GPA/CPA` thiếu trong 1 vài kỳ: impute bằng `forward-fill` theo sinh viên; nếu không có lịch sử, impute bằng median theo lớp/khoá.
* `DIEM_TRUNGTUYEN` / `DIEM_CHUAN` từ admission: nếu thiếu, dùng median theo `NAM_TUYENSINH` hoặc đánh dấu missing với flag.
* Chuẩn hóa categorical: `PTXT`, `TOHOP_XT` → frequency encoding hoặc target encoding. Đối với DL, dùng embedding.

---

# 5) Feature engineering chi tiết

**A. Features tĩnh (per-student)**

* `NAM_TUYENSINH` → năm nhập học (hoặc tuổi khi vào).
* `PTXT`, `TOHOP_XT`, `DIEM_TRUNGTUYEN`, `DIEM_CHUAN`.
* `num_semesters_recorded`, `first_semester`, `last_semester`.

**B. Features chuỗi (per-semester)**

* `GPA`, `CPA`, `TC_DANGKY`, `TC_HOANTHANH`.
* `complete_rate = TC_HOANTHANH / TC_DANGKY` (cap ở [0,1] nếu hợp lý).
* `delta_GPA = GPA_t - GPA_{t-1}`, `delta_complete_rate`.
* `cum_completed_credits` (tích luỹ).
* `semester_gap` nếu có gap (nợ học kỳ, nghỉ).
* `semester_index_normalized` = semester_index - first_semester.

**C. Aggregated snapshot features (dùng cho tree models)**

* last_k_mean(GPA), last_k_std(GPA) (k = 1,2,3).
* slope/trend: fit linear slope của `complete_rate` trên lịch sử (sử dụng np.polyfit).
* weighted_mean (giảm dần theo thời gian) của `complete_rate`.
* ratio: `avg_completed / avg_registered`.
* Count of failing semesters (complete_rate < threshold).

**D. Đặc biệt cho test**: test file chỉ có MA_SO_SV + TC_DANGKY; đảm bảo merge admission + last known semester features before forecasting.

---

# 6) Cách xử lý chuỗi có độ dài khác nhau

Hai trường phái:

**(A) Tree/GBM approach (không giữ chuỗi thô)**

* Aggregate lịch sử thành các statistics (như trên). Ưu: dễ huấn luyện, ít dữ liệu, explainable, robust với missing.
* Nhược: mất thông tin thứ tự tinh vi.

**(B) Sequence models (RNN/Transformer)**

* Tạo tensors: `batch_size x max_seq_len x features`. Pad những sinh viên có ít kỳ và dùng `mask`.
* LSTM/GRU: dùng `pack_padded_sequence` để xử lý hiệu quả.
* Transformer: thêm positional encoding (semester index); xử lý mask cho padding.
* Thêm features tĩnh (admission) bằng cách concat vào đầu hidden state / tile qua các time steps.
* Output: dự báo 1 giá trị cho kỳ tiếp theo (regression).

---

# 7) Mô hình đề xuất & loss/metrics

**Baseline**: `pred = TC_DANGKY * last_complete_rate` hoặc `pred = TC_DANGKY * mean_complete_rate`.

**Tree-based strong baseline**: LightGBM/XGBoost trên aggregated features.

* Loss: MSE (huấn luyện), theo dõi RMSE/MAPE/R^2 trên valid.
* Hyperparams: num_leaves=31–128, learning_rate=0.05–0.2, n_estimators 200–2000, early_stopping.
* Regularization: min_data_in_leaf, lambda_l1/l2.

**Sequence models**:

* LSTM/GRU (1–3 layers), hidden_dim 64–256, dropout 0.2–0.5.
* Transformer encoder: 2–4 layers, heads 4–8.
* Input normalization (per-feature).
* Loss: MSE; you can also try Huber loss or quantile loss to estimate intervals.

**Uncertainty**:

* Quantile regression (LightGBM supports quantile), or MC Dropout for DL.

**Explainability**:

* SHAP for LightGBM (global + per-student).
* For Transformer, visualize attention weights; for LSTM, use Integrated Gradients or attention modules.

---

# 8) Validation & CV (quan trọng vì temporality)

* **Strict time-based split**: train = data up to HK1 2023–2024; valid = HK2 2023–2024 (đề bài yêu cầu). Test = HK1 2024–2025. Không dùng random split. 
* Nếu muốn tuning thêm: time-series cross-validation (rolling origin): train on [t0..tN], valid on tN+1; lặp tăng tN.
* Sử dụng early stopping dựa trên metric RMSE hoặc MAPE trên valid set.

---

# 9) Tư vấn threshold để phát hiện “nguy cơ”

* Tạo `risk_score = 1 - pred / TC_DANGKY`.
* Flag student nếu `pred < alpha * TC_DANGKY` (ví dụ alpha=0.8 hoặc 0.9). Chọn alpha theo trade-off giữa precision & recall trên valid set (vẽ precision-recall cho các threshold).
* Gợi ý hành động: cố vấn học tập liên hệ, giảm TC_DANGKY, tư vấn học tập, hỗ trợ học phí, tutor.

---

# 10) Triển khai & sản phẩm nộp

* Xuất file nộp: `MA_SO_SV, PRED_TC_HOANTHANH` (float) theo định dạng yêu cầu. 
* Báo cáo cần có:

  * Tóm tắt dữ liệu, EDA, chọn features, mô hình, hyperparam, kết quả trên valid (MSE, RMSE, R^2, MAPE), dashboard risk. 
* Code: Jupyter Notebook + script chạy từ đầu tới cuối, requirements.txt, README.

---

# 11) Ví dụ code (skeleton) — LightGBM + LSTM (ý tưởng chính, bạn có thể copy & hoàn thiện)

## A. LightGBM skeleton (Python)

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# 1) load & merge
adm = pd.read_csv("admission.csv")
rec = pd.read_csv("academic_records.csv")
test = pd.read_csv("test.csv")
df = rec.merge(adm, on="MA_SO_SV", how="left")

# 2) build aggregated features per (student, semester) row where semester = target semester
# Example: for each row t, build last_1_complete_rate, mean_last_3, slope...
# (pseudo)
def agg_features(df):
    out = []
    for sid, g in df.groupby("MA_SO_SV"):
        g = g.sort_values("semester_index")
        g['complete_rate'] = g['TC_HOANTHANH']/g['TC_DANGKY']
        # compute rolling
        g['last_complete'] = g['complete_rate'].shift(1)
        g['mean_last_3'] = g['complete_rate'].shift(1).rolling(3, min_periods=1).mean()
        # ... other features
        out.append(g)
    return pd.concat(out)

df_feat = agg_features(df)

# 3) filter train/valid according to semester (time-based)
train = df_feat[df_feat.semester_index <= TRAIN_MAX]
valid = df_feat[df_feat.semester_index == VALID_SEM_INDEX]

features = [c for c in train.columns if c not in ['MA_SO_SV','TC_HOANTHANH','semester_index']]

dtrain = lgb.Dataset(train[features], label=train['TC_HOANTHANH'])
dvalid = lgb.Dataset(valid[features], label=valid['TC_HOANTHANH'])

params = {"objective":"regression","metric":"rmse","learning_rate":0.05,"num_leaves":64}
bst = lgb.train(params, dtrain, valid_sets=[dtrain,dvalid], early_stopping_rounds=100)

pred_valid = bst.predict(valid[features])
print("RMSE:", mean_squared_error(valid['TC_HOANTHANH'], pred_valid, squared=False))
print("R2:", r2_score(valid['TC_HOANTHANH'], pred_valid))
```

## B. LSTM skeleton (PyTorch) — sequence approach

```python
# Prepare padded sequences per student
# X_seq: (N, max_len, feat_dim), mask: (N, max_len), y: target next-term TC_HOANTHANH
# admission static features can be concatenated to final hidden state or tiled across time.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class StudentSeqDataset(Dataset):
    def __init__(self, sequences, seq_lens, static_feats, targets):
        self.sequences = sequences
        self.seq_lens = seq_lens
        self.static = static_feats
        self.targets = targets
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        return (self.sequences[i], self.seq_lens[i], self.static[i]), self.targets[i]

class LSTMReg(nn.Module):
    def __init__(self, input_dim, static_dim, hidden=128, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden + static_dim, 1)
    def forward(self, x_packed, seq_lens, static):
        # use pack_padded_sequence during training loop
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_packed, seq_lens, batch_first=True, enforce_sorted=False)
        out_packed, (h_n, c_n) = self.lstm(packed)
        h_last = h_n[-1]  # (batch, hidden)
        x = torch.cat([h_last, static], dim=1)
        return self.fc(x).squeeze(1)

# Training loop: pack sequences, compute MSELoss, optimizer Adam, scheduler...
```

---

# 12) Tips thực tế & pitfalls

* **Không leak thông tin tương lai**: khi tạo features, đảm bảo bạn chỉ dùng thông tin trước kỳ mục tiêu.
* **Outliers**: nếu TC_HOANTHANH > TC_DANGKY, kiểm tra kỹ (có thể do chuyển đổi tín chỉ khác ngành) — xử lý case-by-case.
* **Small-sample students**: với sinh viên có lịch sử 1 kỳ, sequence models dễ yếu — trộn chiến lược: tuyên bố fallback (dùng baseline) cho các sinh viên có ít lịch sử.
* **Imbalanced coverage of cohorts**: những khoá mới/old có xu hướng khác — add `NAM_TUYENSINH` để model nhận biết.
* **Feature leakage from admission features**: Admission features OK (tiền sử tĩnh), nhưng đừng dùng thông tin từ future semesters.

---

# 13) Giải thích kết quả và đề xuất chính sách

* Dùng SHAP (tree) để xuất top-10 features ảnh hưởng đến việc hoàn thành tín chỉ (ví dụ: last_complete_rate, DIEM_TRUNGTUYEN, trend_complete_rate).
* Nhóm sinh viên “nguy cơ” và gợi ý hành động:

  * Nếu pred < 0.6*TC_DANGKY → urgent outreach (tutor, giảm cc).
  * 0.6–0.8 → monitoring + tư vấn.
  * > 0.8 → bình thường.

---

# 14) Checklist trước nộp bài / báo cáo

* [ ] Confirm split train/valid/test theo đề. 
* [ ] Baseline + LightGBM + 1 sequence model (LSTM/Transformer) + lựa chọn tốt nhất.
* [ ] Bảng kết quả metric (MSE, RMSE, R^2, MAPE) trên valid. 
* [ ] SHAP plots + example case studies (5 sinh viên: đúng/nhầm, phân tích vì sao).
* [ ] Script tạo `ten_doi_thi.csv` (MA_SO_SV, PRED_TC_HOANTHANH).

---

# 15) Nếu bạn muốn — mình sẽ làm tiếp (tự động)

Mình có thể tiếp tục và:

* 1. Viết notebook mẫu (EDA → features → LightGBM full pipeline → SHAP → xuất submit).
* 2. Viết skeleton PyTorch cho LSTM + training loop + packing/padding.
* 3. Viết script `make_submission.py`.

Bạn muốn mình **bắt đầu bằng nào** (ví dụ: notebook LightGBM hoàn chỉnh + mã tạo features)? Mình sẽ cung cấp code chi tiết sẵn để bạn chạy.
