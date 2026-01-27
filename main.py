import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from utils import (
    engineer_features,
    build_splits,
    select_feature_columns,
)

def main():
    """
    Pipeline chính dự báo số tín chỉ hoàn thành (TC_HOANTHANH) cho kỳ HK1 2024-2025.

    Luồng xử lý:
    1) Đọc dữ liệu tuyển sinh, lịch sử học tập, và danh sách test
    2) Tạo đặc trưng theo thời gian (chỉ dùng dữ liệu quá khứ) và hợp nhất đặc trưng tĩnh (admission)
    3) Chia tập theo thời gian: Train ≤ HK1 2023-2024, Valid = HK2 2023-2024
    4) Huấn luyện LightGBM với tham số tốt nhất (nếu có) và đánh giá trên Valid (MSE, RMSE, R^2, MAPE)
    5) Suy luận trên Test, có fallback hợp lý cho SV không có lịch sử, clip dự báo trong [0, TC_DANGKY]
    6) Xuất submission.csv
    """
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")

    adm_path = os.path.join(data_dir, "admission.csv")
    rec_path = os.path.join(data_dir, "academic_records.csv")
    test_path = os.path.join(data_dir, "test.csv")


    admission = pd.read_csv(adm_path)
    records = pd.read_csv(rec_path)
    test = pd.read_csv(test_path)

    # Tạo đặc trưng trên toàn bộ dữ liệu học tập (giai đoạn train+valid)
    # - Bao gồm các đặc trưng lũy tiến, rolling, và giá trị kỳ trước theo từng SV
    # - Merge thêm đặc trưng tĩnh từ dữ liệu tuyển sinh
    df_feat = engineer_features(records, admission)

    # Chia tập theo thời gian nghiêm ngặt theo đề bài
    train_df, valid_df = build_splits(df_feat)
    print("Tách tập theo thời gian: Train ≤ HK1 2023-2024, Valid = HK2 2023-2024")
    # Chọn cột đặc trưng
    feat_cols = select_feature_columns(train_df)

    X_train = train_df[feat_cols].copy()
    y_train = train_df["TC_HOANTHANH"].astype(float)
    
    X_valid = valid_df[feat_cols].copy()
    y_valid = valid_df["TC_HOANTHANH"].astype(float)
    
    # Ép kiểu numeric để tương thích với mô hình (điền thiếu bằng 0)
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_valid = X_valid.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Chuẩn bị thống kê tỷ lệ hoàn thành theo nhóm (cohort) để fallback cho SV mới (không có lịch sử)
    # - Tính completion rate = TC_HOANTHANH / TC_DANGKY trên tập train
    # - Lấy trung bình theo NAM_TUYENSINH, PTXT, TOHOP_XT và trung bình toàn cục
    train_info = train_df[["MA_SO_SV", "TC_DANGKY", "TC_HOANTHANH", "NAM_TUYENSINH"]].merge(
        admission[["MA_SO_SV", "PTXT", "TOHOP_XT"]], on="MA_SO_SV", how="left"
    )
    train_info = train_info[train_info["TC_DANGKY"] > 0].copy()
    train_info["comp_rate"] = (train_info["TC_HOANTHANH"] / train_info["TC_DANGKY"]).clip(0, 1)
    global_mean_rate = float(train_info["comp_rate"].mean()) if not train_info.empty else 0.7
    year_mean = train_info.groupby("NAM_TUYENSINH")["comp_rate"].mean().to_dict()
    ptxt_mean = train_info.groupby("PTXT")["comp_rate"].mean().to_dict()
    toh_mean = train_info.groupby("TOHOP_XT")["comp_rate"].mean().to_dict()

    # Chuẩn bị các mô hình và tham số cơ bản + tham số đã tune (nếu có)
    models_cfg = []

    # LightGBM
    lgb_params = {
        "objective": "regression",
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "random_state": 42,
    }
    tuned_path = os.path.join(root, "best_params_lightgbm.json")
    if os.path.exists(tuned_path):
        with open(tuned_path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        if isinstance(tuned, dict):
            lgb_params.update(tuned)
            print("Đã nạp tham số LightGBM từ best_params_lightgbm.json")
    models_cfg.append(("lightgbm", lgb_params))

    # CatBoost
    cat_params = {
        "loss_function": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_state": 42,
        "verbose": False,
    }
    tuned_path = os.path.join(root, "best_params_catboost.json")
    if os.path.exists(tuned_path):
        with open(tuned_path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        if isinstance(tuned, dict):
            cat_params.update(tuned)
            print("Đã nạp tham số CatBoost từ best_params_catboost.json")
    models_cfg.append(("catboost", cat_params))

    # XGBoost
    xgb_params = {
        "objective": "reg:squarederror",
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
    }
    tuned_path = os.path.join(root, "best_params_xgboost.json")
    if os.path.exists(tuned_path):
        with open(tuned_path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        if isinstance(tuned, dict):
            xgb_params.update(tuned)
            print("Đã nạp tham số XGBoost từ best_params_xgboost.json")
    models_cfg.append(("xgboost", xgb_params))


    # Chuẩn bị dữ liệu test và fallback một lần

    # Tạo đặc trưng test từ lịch sử gần nhất của mỗi sinh viên
    # - Lấy hàng đặc trưng mới nhất cho từng MA_SO_SV, sau đó thay TC_DANGKY bằng giá trị trong test
    last_hist = (
        df_feat.sort_values(["MA_SO_SV", "year_start", "semester_num"]).groupby("MA_SO_SV").tail(1)
    )

    test_merged = test.merge(last_hist[["MA_SO_SV"] + feat_cols], on="MA_SO_SV", how="left")
    # Thay TC_DANGKY bằng giá trị cung cấp trong test
    if "TC_DANGKY" in test.columns:
        test_merged["TC_DANGKY"] = test["TC_DANGKY"].values

    # Thêm thông tin tuyển sinh để nhóm fallback (chỉ PTXT, TOHOP_XT để tránh hậu tố cột)
    test_merged = test_merged.merge(
        admission[["MA_SO_SV", "PTXT", "TOHOP_XT"]], on="MA_SO_SV", how="left"
    )

    # Căn chỉnh cột với tập đặc trưng huấn luyện
    test_X = test_merged[feat_cols].copy()
    test_X = test_X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Tính fallback cho sinh viên mới (không có lịch sử) một lần
    students_with_history = set(df_feat["MA_SO_SV"].unique())
    mask_new = ~test_merged["MA_SO_SV"].isin(students_with_history)
    fallback_pred = None
    if mask_new.any():
        def get_fallback_rate(row):
            year = row.get("NAM_TUYENSINH", np.nan)
            ptxt = row.get("PTXT", np.nan)
            toh = row.get("TOHOP_XT", np.nan)
            if pd.notna(year) and year in year_mean:
                return float(year_mean[year])
            if pd.notna(ptxt) and ptxt in ptxt_mean:
                return float(ptxt_mean[ptxt])
            if pd.notna(toh) and toh in toh_mean:
                return float(toh_mean[toh])
            return global_mean_rate

        fallback_rates = test_merged.loc[mask_new].apply(get_fallback_rate, axis=1).astype(float).values
        fallback_TCD = test_merged.loc[mask_new, "TC_DANGKY"].fillna(0).astype(float).values
        fallback_pred = np.clip(fallback_rates, 0, 1) * fallback_TCD

    # Vòng lặp huấn luyện + đánh giá + dự báo cho từng mô hình
    for model_name, params in models_cfg:
        print(f"\n===== Training {model_name.upper()} =====")
        if model_name == "lightgbm":
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )
        elif model_name == "catboost":
            model = CatBoostRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=100,
                verbose=False,
            )
        elif model_name == "xgboost":
            # Train using native xgboost for consistent early stopping support
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            booster_params = {
                "objective": params.get("objective", "reg:squarederror"),
                "eta": params.get("learning_rate", 0.05),
                "max_depth": params.get("max_depth", 6),
                "min_child_weight": params.get("min_child_weight", 1.0),
                "subsample": params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.8),
                "lambda": params.get("reg_lambda", 1.0),
                "alpha": params.get("reg_alpha", 0.0),
                "tree_method": params.get("tree_method", "hist"),
                "seed": params.get("random_state", 42),
                "verbosity": 0,
                "eval_metric": "rmse",
            }
            num_rounds = int(params.get("n_estimators", 2500))
            booster = xgb.train(
                booster_params,
                dtrain,
                num_boost_round=num_rounds,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=100,
                verbose_eval=False,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Đánh giá trên tập Valid
        if model_name == "xgboost":
            y_pred_valid = booster.predict(xgb.DMatrix(X_valid))
        else:
            y_pred_valid = model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred_valid)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_valid, y_pred_valid)
        denom = np.where(np.abs(y_valid.values) < 1e-6, 1.0, np.abs(y_valid.values))
        mape = np.mean(np.abs((y_valid.values - y_pred_valid) / denom))
        print(f"[{model_name}] Valid MSE:  {mse:.4f}")
        print(f"[{model_name}] Valid RMSE: {rmse:.4f}")
        print(f"[{model_name}] Valid R^2:  {r2:.4f}")
        print(f"[{model_name}] Valid MAPE: {mape:.4f}")

        # Dự báo trên test
        if model_name == "xgboost":
            test_pred = booster.predict(xgb.DMatrix(test_X))
        else:
            test_pred = model.predict(test_X)

        # Áp dụng fallback cho SV mới
        if mask_new.any():
            test_pred[mask_new.values] = fallback_pred

        # Clip dự báo về [0, TC_DANGKY]
        if "TC_DANGKY" in test_merged.columns:
            test_pred = np.minimum(np.maximum(test_pred, 0), test_merged["TC_DANGKY"].values)
        else:
            test_pred = np.maximum(test_pred, 0)

        # Lưu submission riêng cho từng mô hình
        submission = pd.DataFrame({
            "MA_SO_SV": test_merged["MA_SO_SV"],
            "PRED_TC_HOANTHANH": test_pred.astype(float),
        })
        out_path = os.path.join(root, f"submission_{model_name}.csv")
        submission.to_csv(out_path, index=False)
        print(f"Saved submission to: {out_path}")


if __name__ == "__main__":
    # Chạy pipeline
    main()

