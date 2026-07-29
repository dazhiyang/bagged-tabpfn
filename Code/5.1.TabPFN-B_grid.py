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
# TabPFN-B sensitivity (SI): n_boot and M for ONE combo only (yHxP).
#
# Protocol (no 2025 peeking):
#   1. Random 50/50 split of 2024 rows (fixed ENS_SEED).
#   2. Half A = context pool; half B = validation.
#   3. For each (n_boot, M): bootstrap from A, bag M members, score RMSE on B.
#   4. Write Data/Output/TabPFN-B_sensitivity.txt
#
# Production bagging (all combos, full 2024 → 2025) remains Code/2.5.TabPFN-B.py
# with a priori n_boot=2000, M=10.
#
# Grid (sequential, not full product):
#   - Sweep N_BOOT_GRID at fixed M_REF
#   - Sweep M_GRID at fixed N_BOOT_REF (skip duplicate M_REF row)

import os

import numpy as np
import pandas as pd
import sklearn as sk
import tabpfn_client
from tabpfn_client.constants import ModelVersion

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **_kw):  # noqa: ANN001
        return iterable


###############################################################################
# Parameter block
###############################################################################

project_path = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
ENS_SEED = 123

# One (y, x) pair only.
TARGET = "yH"
BASE_FEATURE = "xP"
COMBO = f"{TARGET}{BASE_FEATURE}"
era5_features = ["SZA", "lcc", "mcc", "tcsw", "tcwv"]

N_BOOT_GRID = [500, 1000, 2000, 4000]
M_GRID = [5, 10, 20]
M_REF = 10
N_BOOT_REF = 2000

OUT_SENS = os.path.join(project_path, "Data", "Output", "TabPFN-B_sensitivity.txt")


###############################################################################
# Helpers
###############################################################################


def rmse_wm2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def bagged_predict(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_val: np.ndarray,
    *,
    n_boot: int,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Average M member predictions on X_val; bootstrap contexts from X_ctx."""
    n_a = int(X_ctx.shape[0])
    n_boot_use = min(int(n_boot), n_a)
    preds = []
    for k in range(m):
        idx = rng.integers(low=0, high=n_a, size=n_boot_use)
        model = tabpfn_client.TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5,
            random_state=int(ENS_SEED + k),
        )
        model.fit(np.asarray(X_ctx[idx]), np.asarray(y_ctx[idx]))
        preds.append(np.asarray(model.predict(X_val), dtype=float).ravel())
    return np.mean(np.stack(preds, axis=0), axis=0)


###############################################################################
# Data: 2024 only, random half-split
###############################################################################

file = os.path.join(project_path, "Data", "arranged15min.txt")
df = pd.read_csv(file, sep="\t")
df["Time"] = pd.to_datetime(df["Time"], format="mixed")
df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)
df["SZA"] = np.cos(np.radians(df["SZA"]))

df_2024 = df.loc[df["Time"].dt.year == 2024].copy().reset_index(drop=True)
n_2024 = len(df_2024)
features = [BASE_FEATURE] + era5_features

rng_split = np.random.default_rng(ENS_SEED)
perm = rng_split.permutation(n_2024)
n_a = n_2024 // 2
idx_a = np.sort(perm[:n_a])
idx_b = np.sort(perm[n_a:])

df_a = df_2024.iloc[idx_a].reset_index(drop=True)
df_b = df_2024.iloc[idx_b].reset_index(drop=True)

X_a_raw = df_a[features]
y_a = df_a[TARGET]
X_b_raw = df_b[features]
y_b = df_b[TARGET]
ghc_b = df_b["Ghc"].to_numpy(dtype=float)

scaler = sk.preprocessing.StandardScaler()
X_a = scaler.fit_transform(X_a_raw)
X_b = scaler.transform(X_b_raw)
y_a_np = y_a.to_numpy(dtype=float)
y_b_wm2 = y_b.to_numpy(dtype=float) * ghc_b

print(
    f"TabPFN-B sensitivity (combo={COMBO}): 2024 n={n_2024} → "
    f"context half A n={len(df_a)}, val half B n={len(df_b)}; "
    f"seed={ENS_SEED}. No 2025 used.",
    flush=True,
)

tabpfn_client.set_access_token(tabpfn_client.get_access_token())

###############################################################################
# Grid jobs: (n_boot, M, sweep_tag)
###############################################################################

jobs: list[tuple[int, int, str]] = []
for n_boot in N_BOOT_GRID:
    jobs.append((int(n_boot), int(M_REF), "n_boot"))
for m in M_GRID:
    if int(m) == int(M_REF):
        continue  # already covered at N_BOOT_REF in n_boot sweep if N_BOOT_REF in grid
    jobs.append((int(N_BOOT_REF), int(m), "M"))

# Ensure (N_BOOT_REF, M_REF) appears once even if N_BOOT_REF not in N_BOOT_GRID
if (int(N_BOOT_REF), int(M_REF)) not in {(n, m) for n, m, _ in jobs}:
    jobs.insert(0, (int(N_BOOT_REF), int(M_REF), "n_boot"))

rows: list[dict] = []
# Fresh RNG stream for bootstrap draws (independent of split permutation stream).
rng_boot = np.random.default_rng(ENS_SEED + 10_000)

for n_boot, m, sweep in tqdm(jobs, desc="sensitivity grid", unit="cfg"):
    print(f"Fitting combo={COMBO} n_boot={n_boot} M={m} (sweep={sweep}) …", flush=True)
    y_hat_csi = bagged_predict(
        X_a,
        y_a_np,
        X_b,
        n_boot=n_boot,
        m=m,
        rng=rng_boot,
    )
    y_hat_wm2 = y_hat_csi * ghc_b
    val_rmse = rmse_wm2(y_b_wm2, y_hat_wm2)
    rows.append(
        {
            "combo": COMBO,
            "n_boot": n_boot,
            "M": m,
            "sweep": sweep,
            "n_context_pool": int(len(df_a)),
            "n_val": int(len(df_b)),
            "val_rmse_wm2": round(val_rmse, 4),
            "seed": ENS_SEED,
        }
    )
    print(f"  val RMSE = {val_rmse:.4f} W/m^2", flush=True)

out_df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_SENS), exist_ok=True)
out_df.to_csv(OUT_SENS, sep="\t", index=False)

best = out_df.loc[out_df["val_rmse_wm2"].idxmin()]
print(f"Wrote: {OUT_SENS}", flush=True)
print(
    f"Best on 2024 val (not used for 2.5 defaults): "
    f"n_boot={int(best.n_boot)} M={int(best.M)} "
    f"val_rmse={float(best.val_rmse_wm2):.4f}",
    flush=True,
)
print(
    "Code/2.5.TabPFN-B.py uses n_boot=2000, M=10 "
    "(retained after SI sensitivity vs 2025 test).",
    flush=True,
)
