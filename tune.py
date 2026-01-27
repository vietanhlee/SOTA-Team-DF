import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import xgboost as xgb

from utils import (
    engineer_features,
    build_splits,
    build_splits_last_per_student,
    select_feature_columns,
)
from sklearn.metrics import mean_squared_error


def prepare_data(admission_path: str, records_path: str, test_path: str, split_mode: str = "time"):
    admission = pd.read_csv(admission_path)
    records = pd.read_csv(records_path)
    _ = pd.read_csv(test_path)  # only to ensure path exists; not used in tuning

    df_feat = engineer_features(records, admission)
    if split_mode == "last":
        train_df, valid_df = build_splits_last_per_student(df_feat)
    else:
        train_df, valid_df = build_splits(df_feat)

    feat_cols = select_feature_columns(train_df)
    X_train = train_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = train_df["TC_HOANTHANH"].astype(float)
    X_valid = valid_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_valid = valid_df["TC_HOANTHANH"].astype(float)
    return X_train, y_train, X_valid, y_valid, feat_cols


def suggest_lgb_params(trial: optuna.trial.Trial) -> Dict:
    params = {
        "objective": "regression",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
    }
    max_depth_choice = trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12])
    params["max_depth"] = max_depth_choice
    return params


def suggest_cat_params(trial: optuna.trial.Trial) -> Dict:
    params = {
        "loss_function": "RMSE",
        "random_state": 42,
        "iterations": trial.suggest_int("iterations", 800, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        "allow_writing_files": False,
        "verbose": False,
    }
    return params


def suggest_xgb_params(trial: optuna.trial.Trial) -> Dict:
    params = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "tree_method": trial.suggest_categorical("tree_method", ["hist", "approx"]),
    }
    return params


def objective_factory(model_name: str, X_train, y_train, X_valid, y_valid):
    def objective(trial: optuna.trial.Trial) -> float:
        if model_name == "lightgbm":
            params = suggest_lgb_params(trial)
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
            )
        elif model_name == "catboost":
            params = suggest_cat_params(trial)
            model = CatBoostRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=150,
                verbose=False,
            )
        elif model_name == "xgboost":
            params = suggest_xgb_params(trial)
            # Use native xgboost training for robust early stopping across versions
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
            num_rounds = int(params.get("n_estimators", 1000))
            booster = xgb.train(
                booster_params,
                dtrain,
                num_boost_round=num_rounds,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=150,
                verbose_eval=False,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        # Predict and score
        if model_name == "xgboost":
            y_pred = booster.predict(dvalid)
        else:
            y_pred = model.predict(X_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
        return rmse

    return objective


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    adm_path = os.path.join(data_dir, "admission.csv")
    rec_path = os.path.join(data_dir, "academic_records.csv")
    test_path = os.path.join(data_dir, "test.csv")

    split_mode = os.environ.get("SPLIT_MODE", "time").lower()
    n_trials = int(os.environ.get("N_TRIALS", "20"))

    X_train, y_train, X_valid, y_valid, _ = prepare_data(adm_path, rec_path, test_path, split_mode)

    results = {"split_mode": split_mode, "models": {}, "trials": n_trials}

    # LightGBM study
    lgb_study = optuna.create_study(direction="minimize", study_name=f"lgbm_tuning_{split_mode}")
    lgb_objective = objective_factory("lightgbm", X_train, y_train, X_valid, y_valid)
    t0 = time.time()
    lgb_study.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=False)
    lgb_elapsed = time.time() - t0
    print("[LightGBM] Best RMSE:", lgb_study.best_value)
    print("[LightGBM] Best Params:")
    for k, v in lgb_study.best_params.items():
        print(f"  {k}: {v}")
    print(f"[LightGBM] Elapsed: {lgb_elapsed:.1f}s for {n_trials} trials")
    results["models"]["lightgbm"] = {
        "best_value_rmse": lgb_study.best_value,
        "best_params": lgb_study.best_params,
        "elapsed_sec": lgb_elapsed,
    }
    with open(os.path.join(root, "best_params_lightgbm.json"), "w", encoding="utf-8") as f:
        json.dump(lgb_study.best_params, f, ensure_ascii=False, indent=2)

    # CatBoost study
    cat_study = optuna.create_study(direction="minimize", study_name=f"catboost_tuning_{split_mode}")
    cat_objective = objective_factory("catboost", X_train, y_train, X_valid, y_valid)
    t0 = time.time()
    cat_study.optimize(cat_objective, n_trials=n_trials, show_progress_bar=False)
    cat_elapsed = time.time() - t0
    print("[CatBoost] Best RMSE:", cat_study.best_value)
    print("[CatBoost] Best Params:")
    for k, v in cat_study.best_params.items():
        print(f"  {k}: {v}")
    print(f"[CatBoost] Elapsed: {cat_elapsed:.1f}s for {n_trials} trials")
    results["models"]["catboost"] = {
        "best_value_rmse": cat_study.best_value,
        "best_params": cat_study.best_params,
        "elapsed_sec": cat_elapsed,
    }
    with open(os.path.join(root, "best_params_catboost.json"), "w", encoding="utf-8") as f:
        json.dump(cat_study.best_params, f, ensure_ascii=False, indent=2)

    # XGBoost study
    xgb_study = optuna.create_study(direction="minimize", study_name=f"xgboost_tuning_{split_mode}")
    xgb_objective = objective_factory("xgboost", X_train, y_train, X_valid, y_valid)
    t0 = time.time()
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
    xgb_elapsed = time.time() - t0
    print("[XGBoost] Best RMSE:", xgb_study.best_value)
    print("[XGBoost] Best Params:")
    for k, v in xgb_study.best_params.items():
        print(f"  {k}: {v}")
    print(f"[XGBoost] Elapsed: {xgb_elapsed:.1f}s for {n_trials} trials")
    results["models"]["xgboost"] = {
        "best_value_rmse": xgb_study.best_value,
        "best_params": xgb_study.best_params,
        "elapsed_sec": xgb_elapsed,
    }
    with open(os.path.join(root, "best_params_xgboost.json"), "w", encoding="utf-8") as f:
        json.dump(xgb_study.best_params, f, ensure_ascii=False, indent=2)

    # Save combined results
    with open(os.path.join(root, "tuning_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
