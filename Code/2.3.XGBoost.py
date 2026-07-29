#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
###############################################################################
# This code is co-authored by:
# - Dazhi Yang (yangdazhi.nus@gmail.com)
#   School of Electrical Engineering and Automation,
#   Harbin Institute of Technology (HIT)
# - Yun Chen (PowerPuffYun) (chenyunpku@163.com)
#   Public Meteorological Service Center,
#   China Meteorological Administration (CMA)
###############################################################################
# Same data handling and split as Code/2.1.MLR.R / 2.4.TabPFN.py; predictors =
# retrieval (xS or xP) + explicit ERA5 list. XGBoost + Optuna TPE (independent
# study per combo); writes Data/Output/XGBoost.txt (Time, combo, y, x) and
# Data/Output/XGBoost_best_params.txt (selected hyperparameters + CV MSE).
###############################################################################

import os

import numpy as np
import optuna
import pandas as pd
import sklearn as sk
import xgboost as xgb
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

###############################################################################
# load libraries and set global variables (same layout as ML_models.py)
###############################################################################

# project path
project_path = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"

###############################################################################
# Optuna (TPE): one study per (target × retrieval) combo
###############################################################################
N_TRIALS = int(os.environ.get("XGB_N_TRIALS", "100"))
N_CV = 3
OPTUNA_SEED = 123

###############################################################################
# data handling
###############################################################################

# read the processed file
file = os.path.join(project_path, "Data", "arranged15min.txt")
df = pd.read_csv(file, sep="\t")

# convert the text time to pd time
df["Time"] = pd.to_datetime(df["Time"], format="mixed")

# make irradiances to clear-sky indexes
df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)

# μ₀ = cos(solar zenith angle); column still named "SZA". Matches 2.1.MLR.R; KCDE (2.2) keeps zenith in °.
df["SZA"] = np.cos(np.radians(df["SZA"]))

# chronological split (same as 2.1.MLR.R / 2.2.KCDE.R)
train_year = 2024
test_year = 2025
yt = df["Time"].dt.year
df_train = df.loc[yt == train_year].copy()
df_test = df.loc[yt == test_year].copy()

###############################################################################
# model training and test
###############################################################################

# define targets and features (two sets of observations and two sets of retrievals)
targets = ["yH", "yL"]
# Retrieval order matches Code/2.1.MLR.R / 2.2.KCDE.R: ret = c("xP", "xS") → yHxP, yHxS, yLxP, yLxS blocks.
base_features = [["xP"], ["xS"]]

# Extra predictors beyond xP/xS (same order as cov_use in Code/2.1.MLR.R / era5_features in 2.4.TabPFN.py).
era5_features = ["SZA", "lcc", "mcc", "tcsw", "tcwv"]

out_dir = os.path.join(project_path, "Data", "Output")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "XGBoost.txt")
params_file = os.path.join(out_dir, "XGBoost_best_params.txt")


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def cv_mse(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    model = xgb.XGBRegressor(
        random_state=OPTUNA_SEED,
        n_jobs=1,
        objective="reg:squarederror",
        tree_method="hist",
        **params,
    )
    scores = sk.model_selection.cross_val_score(
        model,
        X,
        y,
        cv=N_CV,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    return float(-scores.mean())


blocks: list[pd.DataFrame] = []
best_rows: list[dict] = []

print(
    f"XGBoost Optuna TPE: n_trials={N_TRIALS}, cv={N_CV}, seed={OPTUNA_SEED} | "
    f"train {train_year} n={len(df_train)}, test {test_year} n={len(df_test)}",
    flush=True,
)

for target in targets:
    for base_feature in base_features:
        features = base_feature + era5_features
        combo = f"{target}{base_feature[0]}"

        # make the design matrix X and target y
        X_train, y_train = df_train[features], df_train[target]
        X_test, y_test = df_test[features], df_test[target]

        # scale the design matrix
        scaler = sk.preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def objective(trial: optuna.Trial) -> float:
            return cv_mse(suggest_params(trial), X_train_scaled, y_train.to_numpy())

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=OPTUNA_SEED),
            study_name=f"xgb_{combo}",
        )
        print(f"{combo}: Optuna optimize ({N_TRIALS} trials) …", flush=True)
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        best_params = dict(study.best_params)
        best_model = xgb.XGBRegressor(
            random_state=OPTUNA_SEED,
            n_jobs=1,
            objective="reg:squarederror",
            tree_method="hist",
            **best_params,
        )
        best_model.fit(X_train_scaled, y_train)

        # predict the clear-sky index
        y_pred = best_model.predict(X_test_scaled)

        # scale back to irradiance
        y_pred_scaled = y_pred * df_test["Ghc"].values
        y_test_scaled = y_test * df_test["Ghc"].values

        time_str = pd.to_datetime(df_test["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        blocks.append(
            pd.DataFrame(
                {
                    "Time": time_str,
                    "combo": combo,
                    "y": np.round(y_test_scaled, 2),
                    "x": np.round(y_pred_scaled, 2),
                }
            )
        )

        rmse_test = float(np.sqrt(np.mean((y_pred_scaled - y_test_scaled) ** 2)))
        row = {
            "combo": combo,
            "cv_mse": study.best_value,
            "test_rmse_wm2": rmse_test,
            "n_trials": N_TRIALS,
            "n_cv": N_CV,
            **best_params,
        }
        best_rows.append(row)
        print(
            f"{combo}: best cv_mse={study.best_value:.6g}  test_RMSE={rmse_test:.2f} W/m2  "
            f"params={best_params}",
            flush=True,
        )

out = pd.concat(blocks, ignore_index=True)
out.to_csv(out_file, sep="\t", index=False)
params_df = pd.DataFrame(best_rows)
params_df.to_csv(params_file, sep="\t", index=False)
print(f"Wrote {out_file} ({len(out)} rows)", flush=True)
print(f"Wrote {params_file} ({len(params_df)} combos)", flush=True)
