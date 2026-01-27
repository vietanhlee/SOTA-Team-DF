Learning Progress Prediction Pipeline

Overview

- Predicts `PRED_TC_HOANTHANH` for each student in `data/test.csv` using time-aware features from `data/academic_records.csv` and static admission info from `data/admission.csv`.
- Time-aware validation with two modes (see below). Saves predictions to `submission.csv`.

Requirements

- Python 3.9+
- Packages: pandas, numpy, scikit-learn, lightgbm, xgboost, optuna

Setup

1. Install dependencies:

- pip install -r requirements.txt

2. Run baseline pipeline (time-based validation):

- python main.py

Validation split modes

- Default global time-based: Train ≤ HK1 2023–2024; Valid = HK2 2023–2024
  - python main.py
- Per-student last-term: Each student’s last available term used for validation
  - On Windows PowerShell: $env:SPLIT_MODE="last"; python main.py
  - On CMD: set SPLIT_MODE=last && python main.py

Full-train (use all data, no validation)

- Trains on all rows and produces submission_full.csv
- python main_fulltrain.py

Hyperparameter tuning (Optuna)

- Optimize LightGBM RMSE on the chosen split. Writes best params to best_params_lightgbm.json and a summary to tuning_results.json.
- Default (time-based split) 20 trials: python tune.py
- Per-student split with 50 trials (PowerShell): $env:SPLIT_MODE="last"; $env:N_TRIALS="50"; python tune.py
- After tuning, rerun python main.py — it auto-loads best_params_lightgbm.json if present.

Grid search (exhaustive)

- Try a bounded grid over strong defaults. Writes best_params_lightgbm.json and grid_tuning_results.json.
- Evaluate all combos: python grid_tune.py
- Sample a subset (PowerShell): $env:MAX_COMBINATIONS="120"; python grid_tune.py
- Per-student split (PowerShell): $env:SPLIT_MODE="last"; python grid_tune.py

What it does

- Parses `HOC_KY` into year/semester, sorts timelines, and builds per-student past-only features (expanding means, cumulative credits, completion ratios, last/rolling completion rates, GPA deltas).
- Encodes admission categorical fields via one-hot.
- Trains a LightGBM regressor with early stopping; reports MSE, RMSE, R^2, MAPE.
- Predicts the test set and clips predictions to [0, TC_DANGKY]. For students with no history, uses cohort-based fallback rates.

Outputs

- submission.csv — columns: MA_SO_SV, PRED_TC_HOANTHANH
- submission_full.csv — full-train variant
- best_params_lightgbm.json — tuned parameters (optional)
- tuning_results.json — tuning run summary (optional)

Notes

- Ensure `data/` contains: `admission.csv`, `academic_records.csv`, `test.csv`.
- If LightGBM installation is problematic on Windows, XGBoost/sklearn GBM can be used for experimentation, but the main pipeline targets LightGBM.
