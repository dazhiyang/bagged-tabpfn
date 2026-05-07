#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Same data pipeline and predictors as Code/2.4.TabPFN.py (tabpfn_client + StandardScaler).
# TabPFN-B (client): one output file per ensemble member m = 1…ENS_K.
# Each member m = 1…ENS_K shares the same TabPFN random_state (ENS_SEED + m − 1).
# Bootstrap indices are redrawn independently for every (target × retrieval) combo fit.
# Each file stacks all combos like TabPFN.txt / MLR.txt (Time, combo, y, x).
# Files: Data/Output/TabPFN-C{m}.txt

import os

import numpy as np
import pandas as pd
import sklearn as sk
import tabpfn_client
from tabpfn_client.constants import ModelVersion

tabpfn_client.set_access_token(tabpfn_client.get_access_token())

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **_kw):  # noqa: ANN001
        return iterable


###############################################################################
# paths
###############################################################################
project_path = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"

###############################################################################
# ensemble (bootstrap bagging): bootstrap_n rows with replacement per combo fit
###############################################################################
ENS_K = 10
ENS_SEED = 123
# None → bootstrap sample size = full training length n_train (standard bagging).
# Integer → m-out-of-n bootstrap (still with replacement), e.g. 2000 for lighter fits.
ENS_BOOTSTRAP_SIZE = 2000

###############################################################################
# data handling (match Code/2.4.TabPFN.py)
###############################################################################

file = os.path.join(project_path, "Data", "arranged15min.txt")
df = pd.read_csv(file, sep="\t")

df["Time"] = pd.to_datetime(df["Time"], format="mixed")

df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)

df["SZA"] = np.cos(np.radians(df["SZA"]))

train_year = 2024
test_year = 2025
yt = df["Time"].dt.year
df_train = df.loc[yt == train_year].copy()
df_test = df.loc[yt == test_year].copy()

n_tr = len(df_train)
bootstrap_n = n_tr if ENS_BOOTSTRAP_SIZE is None else min(int(ENS_BOOTSTRAP_SIZE), n_tr)

print(
    f"TabPFN-B client (tabpfn_client): bagging K={ENS_K}, bootstrap_n={bootstrap_n} "
    f"(with replacement from n_train={n_tr}), seed={ENS_SEED} | "
    f"train {train_year}, test {test_year} n_test={len(df_test)}",
    flush=True,
)

###############################################################################
# model training and test
###############################################################################

targets = ["yH", "yL"]
# Retrieval order matches Code/2.1.MLR.R / 2.2.KCDE.R (xP then xS).
base_features = [["xP"], ["xS"]]
era5_features = ["SZA", "lcc", "mcc", "tcsw", "tcwv"]

out_dir = os.path.join(project_path, "Data", "Output")
os.makedirs(out_dir, exist_ok=True)

rng = np.random.default_rng(ENS_SEED)
combo_jobs = [(t, bf) for t in targets for bf in base_features]
written: list[str] = []

for k in tqdm(range(ENS_K), desc="TabPFN-B client members", unit="member"):
    blocks: list[pd.DataFrame] = []
    for target, base_feature in combo_jobs:
        features = base_feature + era5_features

        X_train, y_train = df_train[features], df_train[target]
        X_test, y_test = df_test[features], df_test[target]

        scaler = sk.preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        combo_tag = f"{target}-{base_feature[0]}"
        combo = f"{target}{base_feature[0]}"
        time_str = pd.to_datetime(df_test["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        y_test_scaled = y_test.to_numpy() * df_test["Ghc"].values

        idx_boot = rng.integers(low=0, high=n_tr, size=bootstrap_n)
        X_sub = np.asarray(X_train_scaled[idx_boot])
        y_sub = np.asarray(y_train.iloc[idx_boot])

        model = tabpfn_client.TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5,
            random_state=int(ENS_SEED + k),
        )
        tqdm.write(f"{combo_tag}: member {k + 1}/{ENS_K} fit (bootstrap_n={bootstrap_n}) …")
        model.fit(X_sub, y_sub)
        X_te = np.asarray(X_test_scaled)
        tqdm.write(f"{combo_tag}: member {k + 1}/{ENS_K} predict …")
        y_k = np.asarray(model.predict(X_te), dtype=float).ravel()

        y_pred_scaled = y_k * df_test["Ghc"].values
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

    out = pd.concat(blocks, ignore_index=True)
    out_file = os.path.join(out_dir, f"TabPFN-C{k + 1}.txt")
    out.to_csv(out_file, sep="\t", index=False)
    written.append(out_file)
    tqdm.write(f"member {k + 1}/{ENS_K}: wrote {out_file} ({len(out)} rows).")

print(f"Wrote {len(written)} files", flush=True)
for fp in written:
    print(f"  {fp}", flush=True)
