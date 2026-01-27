import itertools
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from utils import (
    engineer_features,
    build_splits,
    build_splits_last_per_student,
    select_feature_columns,
)


def prepare_data(admission_path: str, records_path: str, test_path: str, split_mode: str = "time"):
    admission = pd.read_csv(admission_path)
    records = pd.read_csv(records_path)
    _ = pd.read_csv(test_path)  # ensure file exists

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
    return X_train, y_train, X_valid, y_valid


def param_grid() -> Dict[str, List]:
    return {
        "n_estimators": [1500, 2500],
        "learning_rate": [0.03, 0.05],
        "num_leaves": [31, 64, 127],
        "min_child_samples": [20, 80],
        "feature_fraction": [0.80, 0.95],
        "bagging_fraction": [0.80],
        "bagging_freq": [0],
        "reg_lambda": [0.0, 1.0],
        "reg_alpha": [0.0],
        "max_depth": [-1, 8],
    }


def grid_combinations(grid: Dict[str, List]) -> List[Dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def evaluate_combo(params: Dict, X_train, y_train, X_valid, y_valid) -> float:
    final_params = {
        "objective": "regression",
        "random_state": 42,
        **params,
    }
    model = lgb.LGBMRegressor(**final_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )
    pred = model.predict(X_valid)
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    return rmse


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    adm_path = os.path.join(data_dir, "admission.csv")
    rec_path = os.path.join(data_dir, "academic_records.csv")
    test_path = os.path.join(data_dir, "test.csv")

    split_mode = os.environ.get("SPLIT_MODE", "time").lower()
    max_eval_env = os.environ.get("MAX_COMBINATIONS")
    max_combos = int(max_eval_env) if max_eval_env and max_eval_env.isdigit() else None

    X_train, y_train, X_valid, y_valid = prepare_data(adm_path, rec_path, test_path, split_mode)

    grid = param_grid()
    combos = grid_combinations(grid)
    total = len(combos)
    if max_combos is not None and max_combos < total:
        random.Random(42).shuffle(combos)
        combos = combos[:max_combos]
        print(f"Sampling {len(combos)} of {total} grid combinations (MAX_COMBINATIONS={max_combos}).")
    else:
        print(f"Evaluating all {total} grid combinations.")

    best_rmse = float("inf")
    best_params = None
    history: List[Tuple[float, Dict]] = []

    t0 = time.time()
    for i, p in enumerate(combos, start=1):
        rmse = evaluate_combo(p, X_train, y_train, X_valid, y_valid)
        history.append((rmse, p))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = p
        if i % 10 == 0 or i == len(combos):
            print(f"[{i}/{len(combos)}] current best RMSE={best_rmse:.5f}")

    elapsed = time.time() - t0
    print("Best RMSE:", best_rmse)
    print("Best Params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Elapsed: {elapsed:.1f}s for {len(combos)} combos (mode={split_mode})")

    # Save outputs
    with open(os.path.join(root, "best_params_lightgbm.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    with open(os.path.join(root, "grid_tuning_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_mode": split_mode,
                "best_value_rmse": best_rmse,
                "best_params": best_params,
                "evaluated": len(combos),
                "elapsed_sec": elapsed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
