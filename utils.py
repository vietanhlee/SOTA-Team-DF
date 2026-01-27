from typing import Tuple, List
import re
import numpy as np
import pandas as pd

def parse_hoc_ky(hk: str) -> Tuple[int, int]:
    """Phân tích chuỗi HOC_KY thành (năm_bắt_đầu, số_kỳ).

    Định dạng kỳ vọng:
    - "HK1 2020-2021"
    - "HK2 2023-2024"

    Trả về bộ (năm_bắt_đầu, số_kỳ). Nếu không phân tích được, trả về (np.nan, np.nan).
    """
    if not isinstance(hk, str):
        return (np.nan, np.nan)
    m = re.match(r"^HK([12])\s+(\d{4})-(\d{4})$", hk.strip())
    if m:
        sem = int(m.group(1))
        year_start = int(m.group(2))
        return (year_start, sem)
    return (np.nan, np.nan)

def build_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (train_df, valid_df) theo tách thời gian:
    - Train: đến HK1 2023-2024 (year_start < 2023 hoặc year_start == 2023 & semester_num == 1)
    - Valid: HK2 2023-2024 (year_start == 2023 & semester_num == 2)
    """
    train_mask = (df["year_start"] < 2023) | ((df["year_start"] == 2023) & (df["semester_num"] == 1))
    valid_mask = (df["year_start"] == 2023) & (df["semester_num"] == 2)
    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    return train_df, valid_df

def build_splits_last_per_student(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chia theo 'kỳ cuối của mỗi sinh viên' làm valid.

    Ý tưởng:
    - Với mỗi `MA_SO_SV`, chọn bản ghi có `semester_index` lớn nhất làm VALID (kỳ cuối cùng của SV đó).
    - Các bản ghi còn lại của SV đó thuộc TRAIN.

    Lưu ý:
    - Cách này đảm bảo đánh giá 'dự báo kỳ tiếp theo' ở mức per-student, nhưng trộn lẫn mốc thời gian thực giữa các SV.
    - Nếu một SV chỉ có 1 bản ghi, bản ghi đó sẽ vào VALID (TRAIN không có mẫu cho SV đó). Điều này thường vẫn ổn vì TRAIN tổng hợp từ nhiều SV khác.
    """
    # Giả định dữ liệu đã chuẩn, nếu thiếu 'semester_index' thì tự tính từ thời gian
    if "semester_index" not in df.columns:
        df = df.sort_values(["MA_SO_SV", "year_start", "semester_num"]).copy()
        df["semester_index"] = df.groupby("MA_SO_SV").cumcount() + 1

    # Tìm index của dòng cuối cùng cho mỗi sinh viên
    last_idx = (
        df.sort_values(["MA_SO_SV", "year_start", "semester_num"])\
          .groupby("MA_SO_SV").tail(1).index
    )

    valid_df = df.loc[last_idx].copy()
    train_df = df.drop(index=last_idx).copy()
    return train_df, valid_df

def align_columns(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Căn chỉnh cột của `other` theo `base`, điền thiếu bằng 0."""
    return other.reindex(columns=base.columns, fill_value=0)

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Chọn các cột đặc trưng, loại bỏ định danh và nhãn."""
    exclude = {
        "MA_SO_SV",
        "HOC_KY",
        "TC_HOANTHANH",
        "year_start",
        "semester_num",
        "semester_index",
        "prev_TCD",
        "prev_TCH",
        "prev_GPA",
        "prev_GPA2",
    }
    return [c for c in df.columns if c not in exclude]


def engineer_features(records: pd.DataFrame,
                      admission: pd.DataFrame) -> pd.DataFrame:
    """Gộp dữ liệu tuyển sinh (admission) vào lịch sử học tập (records) và tạo đặc trưng theo thời gian.

    Nguyên tắc: chỉ sử dụng dữ liệu QUÁ KHỨ cho đặc trưng của một kỳ, tránh leakage.

    Danh sách đặc trưng tạo ra:
    - past_avg_gpa, past_avg_cpa: trung bình lũy tiến của GPA/CPA các kỳ trước
    - past_total_dangky, past_total_hoanthanh: tổng lũy tiến tín chỉ đã đăng ký/hoàn thành các kỳ trước
    - past_ratio_hoanthanh: tỷ lệ hoàn thành lũy tiến (clip trong [0,1])
    - past_last_complete_rate: tỷ lệ hoàn thành của kỳ ngay trước (clip trong [0,1])
    - last_3_mean_complete_rate: trung bình trượt 3 kỳ gần nhất của past_last_complete_rate
    - past_delta_gpa, past_last_gpa: biến động GPA kỳ gần nhất và GPA kỳ trước
    - num_prev_semesters: số kỳ đã học trước đó (semester_index - 1)
    - diff_diem: mức chênh điểm trúng tuyển so với điểm chuẩn
    - One-hot: PTXT, TOHOP_XT (giữ cả giá trị thiếu)
    """
    df = records.copy()
    # Chuẩn hóa cột thời gian
    df[["year_start", "semester_num"]] = df["HOC_KY"].apply(parse_hoc_ky).apply(pd.Series)
    df = df.sort_values(["MA_SO_SV", "year_start", "semester_num"]).reset_index(drop=True)
    df["semester_index"] = df.groupby("MA_SO_SV").cumcount() + 1

    # Hợp nhất đặc trưng tĩnh từ tuyển sinh
    adm = admission.copy()
    # Làm sạch cơ bản cho DIEM_TRUNGTUYEN/DIEM_CHUAN bằng median theo NAM_TUYENSINH
    for col in ["DIEM_TRUNGTUYEN", "DIEM_CHUAN"]:
        med_per_year = adm.groupby("NAM_TUYENSINH")[col].transform(
            lambda s: s.fillna(s.median())
        )
        adm[col] = adm[col].fillna(med_per_year)

    df = df.merge(adm, on="MA_SO_SV", how="left")

    # Đặc trưng tĩnh: chênh lệch điểm so với chuẩn
    df["diff_diem"] = df["DIEM_TRUNGTUYEN"].fillna(0) - df["DIEM_CHUAN"].fillna(0)
    
    # Mã hóa one-hot cho các cột phân loại từ tuyển sinh
    cat_cols: List[str] = []
    for c in ["PTXT", "TOHOP_XT"]:
        cat_cols.append(c)
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=True, prefix=cat_cols)

    # Đặc trưng theo nhóm sinh viên, chỉ dùng dữ liệu quá khứ
    g = df.groupby("MA_SO_SV", group_keys=False)

    # Các giá trị kỳ trước (dịch theo thời gian)
    df["prev_TCD"] = g["TC_DANGKY"].shift(1)
    df["prev_TCH"] = g["TC_HOANTHANH"].shift(1)
    df["prev_GPA"] = g["GPA"].shift(1)
    df["prev_GPA2"] = g["GPA"].shift(2)

    # Tỷ lệ hoàn thành từ các kỳ trước
    df["past_last_complete_rate"] = (df["prev_TCH"] / df["prev_TCD"]).replace([np.inf, -np.inf], np.nan)
    df["past_last_complete_rate"] = df["past_last_complete_rate"].clip(lower=0, upper=1)

    def expanding_mean_shift(series: pd.Series) -> pd.Series:
        s = series.shift(1)
        return s.expanding().mean()

    def expanding_cumsum_shift(series: pd.Series) -> pd.Series:
        s = series.shift(1)
        return s.cumsum()

    df["past_avg_gpa"] = g["GPA"].apply(expanding_mean_shift).reset_index(level=0, drop=True)
    df["past_avg_cpa"] = g["CPA"].apply(expanding_mean_shift).reset_index(level=0, drop=True)
    df["past_total_dangky"] = g["TC_DANGKY"].apply(expanding_cumsum_shift).reset_index(level=0, drop=True)
    df["past_total_hoanthanh"] = g["TC_HOANTHANH"].apply(expanding_cumsum_shift).reset_index(level=0, drop=True)
    df["past_ratio_hoanthanh"] = (df["past_total_hoanthanh"] / df["past_total_dangky"]).replace([np.inf, -np.inf], np.nan)
    df["past_ratio_hoanthanh"] = df["past_ratio_hoanthanh"].clip(lower=0, upper=1)

    # Tỷ lệ hoàn thành trung bình trượt 3 kỳ (chỉ quá khứ)
    complete_prev = df["past_last_complete_rate"]
    df["last_3_mean_complete_rate"] = g["past_last_complete_rate"].apply(lambda s: s.rolling(3).mean()).reset_index(level=0, drop=True)
    df["last_3_mean_complete_rate"] = df["last_3_mean_complete_rate"].clip(lower=0, upper=1)

    # (Đã bỏ EMA và slope để giữ mô hình đơn giản)

    # Độ chênh GPA giữa các kỳ gần nhất
    df["past_delta_gpa"] = (df["prev_GPA"] - df["prev_GPA2"]).fillna(0)
    df["past_last_gpa"] = df["prev_GPA"].fillna(0)

    # Đếm số kỳ đã học trước đó
    df["num_prev_semesters"] = df["semester_index"].fillna(1) - 1

    # Thay thế giá trị thiếu còn lại trong các đặc trưng đã tạo
    num_cols = [
        "past_avg_gpa",
        "past_avg_cpa",
        "past_total_dangky",
        "past_total_hoanthanh",
        "past_ratio_hoanthanh",
        "past_last_complete_rate",
        "last_3_mean_complete_rate",

        "past_delta_gpa",
        "past_last_gpa",
        "num_prev_semesters",
        "diff_diem",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Không thêm tương tác đặc trưng để mô hình đơn giản và ổn định

    return df
