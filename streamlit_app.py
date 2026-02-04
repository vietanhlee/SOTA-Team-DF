import io
import hashlib
import os
import pickle

import numpy as np
import pandas as pd
import shap
import streamlit as st

from pipeline import FEATURES, build_main_df, build_prediction_frame, predict_credits, train_xgb_regressor


st.set_page_config(page_title="Learning Progress Prediction", layout="wide")


def _fingerprint_frame(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    meta = ("|".join(map(str, df.columns))).encode("utf-8") + b"||" + str(tuple(df.shape)).encode("utf-8")
    return hashlib.sha256(meta + b"||" + hashed).hexdigest()


def _load_or_compute_shap_values(
    model,
    ready_df: pd.DataFrame,
    cache_path: str = "shap_xgb.pkl",
) -> np.ndarray:
    fp = _fingerprint_frame(ready_df)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("fingerprint") == fp:
                return payload["shap_values"]
        except Exception:
            # If cache is corrupt or incompatible, fall back to recompute.
            pass

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(ready_df)
    payload = {
        "fingerprint": fp,
        "x_shape": tuple(ready_df.shape),
        "feature_names": list(ready_df.columns),
        "expected_value": explainer.expected_value,
        "shap_values": shap_values,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    return shap_values


@st.cache_resource
def _load_training_artifacts():
    academic_df = pd.read_csv("data/academic_records.csv")
    admission_df = pd.read_csv("data/admission.csv")

    main_df = build_main_df(academic_df, admission_df)
    model = train_xgb_regressor(main_df, best_params_path="best_params_xgboost.json")

    explainer = shap.TreeExplainer(model)
    return main_df, model, explainer


def _format_shap_note(
    row_features: pd.Series,
    shap_values_row: np.ndarray,
    top_k: int,
    tc_dangky: float,
) -> str:
    shap_series = pd.Series(shap_values_row, index=row_features.index)
    shap_series = shap_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    top = shap_series.abs().sort_values(ascending=False).head(top_k)

    lines = [
        "Sự ảnh hưởng của các đặc trưng dữ liệu đến kết quả dự đoán như sau:",
        f"TC_DANGKY = {tc_dangky:g} là hệ số nhân để quy đổi PASS_RATE → số tín chỉ hoàn thành.",
    ]

    for feature in top.index:
        val = float(shap_series.loc[feature])
        direction = "tăng" if val >= 0 else "giảm"
        lines.append(f"{feature} ảnh hưởng {direction} {abs(val):.6f}")

    return "\n".join(lines)


st.title("Dự đoán số tín chỉ hoàn thành (XGBoost + SHAP)")

with st.sidebar:
    st.header("Thiết lập")
    top_k = st.slider("Số feature SHAP hiển thị", min_value=3, max_value=20, value=8, step=1)
    preferred_last_hk = st.text_input("Snapshot học kỳ gần nhất", value="HK2 2023-2024")

st.markdown(
    "Upload CSV giống `data/test.csv` (tối thiểu cần `MA_SO_SV` và `TC_DANGKY`, `HOC_KY` là tuỳ chọn)."
)

uploaded = st.file_uploader("Chọn file CSV", type=["csv"])  # noqa: E501

if uploaded is None:
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Không đọc được CSV: {e}")
    st.stop()

st.subheader("Dữ liệu đầu vào")
st.dataframe(uploaded_df.head(200), use_container_width=True)

# Load model + base data
try:
    main_df, model, explainer = _load_training_artifacts()
except Exception as e:
    st.error(f"Không load được dữ liệu/mô hình: {e}")
    st.stop()

# Build features for prediction
try:
    ready_df, pred_df = build_prediction_frame(uploaded_df, main_df, preferred_last_hk=preferred_last_hk)
except Exception as e:
    st.error(str(e))
    st.stop()

# Predict
pass_rate, pred_tc = predict_credits(model, ready_df, tc_dangky=pred_df.get("TC_DANGKY_x", pred_df.get("TC_DANGKY", 0)).values)

out_df = uploaded_df.copy()

out_df["PRED_PASS_RATE"] = pass_rate
out_df["PRED_TC_HOANTHANH"] = pred_tc

# SHAP explanation
with st.spinner("Đang tính SHAP để tạo cột ghi chú..."):
    shap_values = _load_or_compute_shap_values(model, ready_df, cache_path="shap_xgb.pkl")

notes = []
for i in range(len(ready_df)):
    tc_val = float(pred_df.get("TC_DANGKY_x", pred_df.get("TC_DANGKY", 0)).iloc[i])
    notes.append(_format_shap_note(ready_df.iloc[i], shap_values[i], top_k=top_k, tc_dangky=tc_val))

out_df["GHI_CHU_SHAP"] = notes

st.subheader("Kết quả dự đoán")
st.dataframe(out_df.head(200), use_container_width=True)

# Download
csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Tải file CSV kết quả",
    data=csv_bytes,
    file_name="prediction_with_shap_notes.csv",
    mime="text/csv",
)

st.caption(f"Model dùng {len(FEATURES)} features. SHAP giải thích cho dự báo PASS_RATE (tỷ lệ hoàn thành).")
