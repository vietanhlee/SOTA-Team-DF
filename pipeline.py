import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


FEATURES: List[str] = [
    "TC_DANGKY",
    "SO_NAM_HOC",
    "COVID_ONLINE_THPT",
    "COVID_ONLINE_DH",
    "HIST_AVG_GPA",
    "HIST_AVG_TC_DANGKY",
    "HIST_PASS_RATE",
    "HIST_GPA_STD",
    "TOTAL_FAIL_CREDITS",
    "LAG1_GPA",
    "LAG2_GPA",
    "LAG1_PASS_RATE",
    "TREND_GPA",
    "LOAD_RATIO",
    "SCORE_GAP",
    "DIEM_TRUNGTUYEN",
    "DIEM_CHUAN",
    "NAM_TUYENSINH",
    "PTXT_1",
    "PTXT_100",
    "PTXT_200",
    "PTXT_3",
    "PTXT_402",
    "PTXT_409",
    "PTXT_5",
    "PTXT_500",
    "AVG_SCORE",
]


def _get_current_year(hoc_ky: str) -> int:
    arr = str(hoc_ky).split("-")
    return int(arr[-1])


def _key_hoc_ky(hoc_ky: str) -> Tuple[int, int]:
    hk, year = str(hoc_ky).split()
    hk = int(hk.replace("HK", ""))
    year = int(year.split("-")[0])
    return (year, hk)


def _latest_semester_label(values: Iterable[str]) -> Optional[str]:
    labels = [v for v in pd.Series(list(values)).dropna().astype(str).unique().tolist()]
    if not labels:
        return None
    labels_sorted = sorted(labels, key=_key_hoc_ky)
    return labels_sorted[-1]


@dataclass(frozen=True)
class TrainingArtifacts:
    main_df: pd.DataFrame
    model: XGBRegressor


def build_main_df(academic_df: pd.DataFrame, admission_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the engineered dataset (close to the notebook logic).

    Output contains one row per (MA_SO_SV, HOC_KY) with engineered history features.
    """

    df = pd.merge(academic_df, admission_df, how="left", on="MA_SO_SV")

    # One-hot PTXT (keep stable set as much as possible)
    if "PTXT" in df.columns:
        dummies = pd.get_dummies(df["PTXT"], prefix="PTXT", dtype=float)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=["PTXT"])

    df["SO_NAM_HOC"] = df["HOC_KY"].apply(_get_current_year) - df["NAM_TUYENSINH"]

    df["_key_hocky"] = df["HOC_KY"].map(_key_hoc_ky)
    df = df.sort_values(by=["MA_SO_SV", "_key_hocky"]).drop(columns=["_key_hocky"])

    # History aggregates
    df["HIST_AVG_GPA"] = (
        df.groupby("MA_SO_SV")["GPA"].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    )
    df["HIST_AVG_TC_DANGKY"] = (
        df.groupby("MA_SO_SV")["TC_DANGKY"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0)
    )

    df["HIST_PASS_RATE"] = (
        df.groupby("MA_SO_SV")["TC_HOANTHANH"].cumsum().shift(1)
        / df.groupby("MA_SO_SV")["TC_DANGKY"].cumsum().shift(1)
    ).fillna(1)

    df["LAG1_GPA"] = df.groupby("MA_SO_SV")["GPA"].shift(1).fillna(0)
    df["LAG2_GPA"] = df.groupby("MA_SO_SV")["GPA"].shift(2).fillna(0)
    df["LAG1_PASS_RATE"] = df.groupby("MA_SO_SV")["HIST_PASS_RATE"].shift(1).fillna(1)
    df["TREND_GPA"] = (df["LAG1_GPA"] - df["HIST_AVG_GPA"]).fillna(0)

    df["PASS_RATE"] = (df["TC_HOANTHANH"] / df["TC_DANGKY"]).replace([np.inf, -np.inf], np.nan)
    df["PASS_RATE"] = df["PASS_RATE"].fillna(0).astype(float)

    df["SCORE_GAP"] = df["DIEM_TRUNGTUYEN"] - df["DIEM_CHUAN"]

    df["HIST_GPA_STD"] = (
        df.groupby("MA_SO_SV")["GPA"].transform(lambda x: x.shift(1).expanding().std()).fillna(0)
    )

    df["FAIL_CREDITS_TEMP"] = df["TC_DANGKY"] - df["TC_HOANTHANH"]
    df["TOTAL_FAIL_CREDITS"] = (
        df.groupby("MA_SO_SV")["FAIL_CREDITS_TEMP"].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    )
    df = df.drop(columns=["FAIL_CREDITS_TEMP"])

    df["LOAD_RATIO"] = df["TC_DANGKY"] / df["HIST_AVG_TC_DANGKY"].replace(0, 1)

    avg_score_map = {
        2025: 6.50,
        2024: 6.57,
        2023: 6.03,
        2022: 6.34,
        2021: 4.97,
        2020: 5.19,
        2019: 4.30,
        2018: 3.79,
        2017: 4.59,
        2016: 4.50,
        2015: 5.25,
    }
    df["AVG_SCORE"] = df["NAM_TUYENSINH"].map(avg_score_map)

    covid_years_thpt = [2020, 2021]
    covid_semesters_dh = [
        "HK1 2019-2020",
        "HK2 2019-2020",
        "HK1 2020-2021",
        "HK2 2020-2021",
        "HK1 2021-2022",
    ]
    df["COVID_ONLINE_THPT"] = df["NAM_TUYENSINH"].isin(covid_years_thpt).astype(int)
    df["COVID_ONLINE_DH"] = df["HOC_KY"].isin(covid_semesters_dh).astype(int)

    # Ensure any missing PTXT dummy columns exist for downstream.
    for col in [c for c in FEATURES if c.startswith("PTXT_")]:
        if col not in df.columns:
            df[col] = 0.0

    # Fill other feature columns if missing
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df


def train_xgb_regressor(main_df: pd.DataFrame, best_params_path: str) -> XGBRegressor:
    best_params = {}
    if os.path.exists(best_params_path):
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)

    X = main_df[FEATURES].copy()
    y = main_df["PASS_RATE"].astype(float).copy()

    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        **best_params,
    )
    model.fit(X, y)
    return model


def build_prediction_frame(
    uploaded_df: pd.DataFrame,
    main_df: pd.DataFrame,
    preferred_last_hk: str = "HK2 2023-2024",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build (ready_df, pred_df)

    - uploaded_df is expected to have: MA_SO_SV, TC_DANGKY, (optional) HOC_KY
    - pred_df is the merged dataframe used for output columns.
    - ready_df is feature-only dataframe aligned to FEATURES.
    """

    if "MA_SO_SV" not in uploaded_df.columns or "TC_DANGKY" not in uploaded_df.columns:
        raise ValueError("CSV upload phải có cột MA_SO_SV và TC_DANGKY.")

    last_hk = preferred_last_hk
    if last_hk not in set(main_df["HOC_KY"].astype(str).unique().tolist()):
        inferred = _latest_semester_label(main_df["HOC_KY"].astype(str).values)
        if inferred is None:
            raise ValueError("Không xác định được học kỳ gần nhất từ dữ liệu lịch sử.")
        last_hk = inferred

    last_snapshot = main_df[main_df["HOC_KY"].astype(str) == str(last_hk)].copy()

    # Merge: uploaded test-like data + last snapshot
    pred_df = pd.merge(uploaded_df, last_snapshot, how="left", on="MA_SO_SV", suffixes=("_x", "_y"))

    # Update SO_NAM_HOC for next term
    if "SO_NAM_HOC" in pred_df.columns:
        pred_df["SO_NAM_HOC"] = pd.to_numeric(pred_df["SO_NAM_HOC"], errors="coerce").fillna(0) + 1
    else:
        pred_df["SO_NAM_HOC"] = 0

    # Map pred features
    pred_features = [
        "TC_DANGKY_x",
        "SO_NAM_HOC",
        "COVID_ONLINE_THPT",
        "COVID_ONLINE_DH",
        "HIST_AVG_GPA",
        "HIST_AVG_TC_DANGKY",
        "HIST_PASS_RATE",
        "HIST_GPA_STD",
        "TOTAL_FAIL_CREDITS",
        "LAG1_GPA",
        "LAG2_GPA",
        "LAG1_PASS_RATE",
        "TREND_GPA",
        "LOAD_RATIO",
        "SCORE_GAP",
        "DIEM_TRUNGTUYEN",
        "DIEM_CHUAN",
        "NAM_TUYENSINH",
        "PTXT_1",
        "PTXT_100",
        "PTXT_200",
        "PTXT_3",
        "PTXT_402",
        "PTXT_409",
        "PTXT_5",
        "PTXT_500",
        "AVG_SCORE",
    ]

    for col in pred_features:
        if col not in pred_df.columns:
            pred_df[col] = 0

    ready_df = pred_df[pred_features].rename(columns={"TC_DANGKY_x": "TC_DANGKY"})

    # Align to FEATURES
    ready_df = ready_df.reindex(columns=FEATURES, fill_value=0)

    # Numeric safety
    for col in FEATURES:
        ready_df[col] = pd.to_numeric(ready_df[col], errors="coerce").fillna(0)

    return ready_df, pred_df

def predict_credits(model: XGBRegressor, ready_df: pd.DataFrame, tc_dangky: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass_rate = model.predict(ready_df)
    pass_rate = np.clip(pass_rate, 0.0, 1.0)
    tc = np.asarray(tc_dangky, dtype=float)
    tc = np.nan_to_num(tc, nan=0.0, posinf=0.0, neginf=0.0)
    pred_tc = tc * pass_rate
    return pass_rate, pred_tc
