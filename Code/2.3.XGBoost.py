#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Same data handling and split as Code/2.1.MLR.R / 2.4.TabPFN.py; predictors =
# retrieval (xS or xP) + explicit ERA5 list. XGBoost + GridSearchCV; writes
# Data/Output/XGBoost.txt (Time, combo, y, x).
###############################################################################

import os

import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb

###############################################################################
# load libraries and set global variables (same layout as ML_models.py)
###############################################################################

# project path
project_path = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"

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

# Core boosting hyperparameters only (XGBoost defaults for row/col subsample,
# min_child_weight, reg_lambda, reg_alpha). Spans low→high trees, shallow→deep
# trees, and small→large step size (≈ geometric in η).
xgb_param_grid = {
    "n_estimators": [200, 450, 800, 1200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.02, 0.05, 0.12, 0.25],
}

out_dir = os.path.join(project_path, "Data", "Output")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "XGBoost.txt")

blocks: list[pd.DataFrame] = []

for target in targets:
    for base_feature in base_features:
        features = base_feature + era5_features

        # make the design matrix X and target y
        X_train, y_train = df_train[features], df_train[target]
        X_test, y_test = df_test[features], df_test[target]

        # scale the design matrix
        scaler = sk.preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBRegressor(
            random_state=123,
            n_jobs=1,
            objective="reg:squarederror",
            tree_method="hist",
        )

        grid_search = sk.model_selection.GridSearchCV(
            model,
            xgb_param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        # predict the clear-sky index
        y_pred = best_model.predict(X_test_scaled)

        # scale back to irradiance
        y_pred_scaled = y_pred * df_test["Ghc"].values
        y_test_scaled = y_test * df_test["Ghc"].values

        time_str = pd.to_datetime(df_test["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        combo = f"{target}{base_feature[0]}"

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

        print(
            f"{target}-{base_feature[0]} using XGBoost trained with params: "
            f"{grid_search.best_params_}"
        )

out = pd.concat(blocks, ignore_index=True)
out.to_csv(out_file, sep="\t", index=False)
print(f"Wrote {out_file} ({len(out)} rows)")
