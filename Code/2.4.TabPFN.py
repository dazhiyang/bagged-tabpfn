#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Same data pipeline as Code/2.3.XGBoost.py (train/test years, scaler).
# Predictors = retrieval + explicit ERA5 columns (cov_use in Code/2.1.MLR.R).
# TabPFNRegressor() uses tabpfn_client defaults only. Writes Data/Output/TabPFN.txt.

import os

import numpy as np
import pandas as pd
import sklearn as sk
import tabpfn_client

tabpfn_client.set_access_token(tabpfn_client.get_access_token())

###############################################################################
# paths
###############################################################################
project_path = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"

###############################################################################
# data handling
###############################################################################

file = os.path.join(project_path, "Data", "arranged15min.txt")
df = pd.read_csv(file, sep="\t")

df["Time"] = pd.to_datetime(df["Time"], format="mixed")

df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)

# μ₀ = cos(solar zenith angle); column still named "SZA". Matches 2.1.MLR.R / 2.3.XGBoost.py; KCDE keeps °.
df["SZA"] = np.cos(np.radians(df["SZA"]))

train_year = 2024
test_year = 2025
yt = df["Time"].dt.year
df_train = df.loc[yt == train_year].copy()
df_test = df.loc[yt == test_year].copy()

# Full calendar years — no random subsampling (contrast e.g. train_test_split scripts).
print(f"TabPFN client: using ALL rows — train {train_year} n={len(df_train)}, test {test_year} n={len(df_test)}")

###############################################################################
# model training and test
###############################################################################

targets = ["yH", "yL"]
# Retrieval order matches Code/2.1.MLR.R / 2.2.KCDE.R (xP then xS).
base_features = [["xP"], ["xS"]]

# Extra predictors beyond xP/xS (keep in sync with cov_use in Code/2.1.MLR.R / era5_features in 2.3.XGBoost.py).
era5_features = ["SZA", "lcc", "mcc", "tcsw", "tcwv"]

out_dir = os.path.join(project_path, "Data", "Output")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "TabPFN.txt")

model = tabpfn_client.TabPFNRegressor()
blocks: list[pd.DataFrame] = []


for target in targets:
    for base_feature in base_features:
        features = base_feature + era5_features

        X_train, y_train = df_train[features], df_train[target]
        X_test, y_test = df_test[features], df_test[target]

        scaler = sk.preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(
            f"{target}-{base_feature[0]}: fit n_train={X_train_scaled.shape[0]}, "
            f"predict n_test={X_test_scaled.shape[0]}"
        )

        model.fit(X_train_scaled, y_train)
        y_pred = np.asarray(model.predict(X_test_scaled), dtype=float).ravel()

        y_pred_scaled = y_pred * df_test["Ghc"].values
        y_test_scaled = y_test.to_numpy() * df_test["Ghc"].values

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

        print(f"{target}-{base_feature[0]} TabPFN client finished")

out = pd.concat(blocks, ignore_index=True)
out.to_csv(out_file, sep="\t", index=False)
print(f"Wrote {out_file} ({len(out)} rows)")
