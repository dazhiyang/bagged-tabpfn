#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
#################################################################################
# This code is co-authored by:
# - Dazhi Yang (yangdazhi.nus@gmail.com)
#   School of Electrical Engineering and Automation,
#   Harbin Institute of Technology (HIT)
# - Yun Chen (PowerPuffYun) (chenyunpku@163.com)
#   Public Meteorological Service Center,
#   China Meteorological Administration (CMA)
#################################################################################

1.2 XAI — SHAP (TreeExplainer) results for **XGBoost**, combo **yHxP** (no figures).

Same statistical setup as Code/2.3.XGBoost.py: clear-sky indices, cos(SZA),
train 2024 / explain on 2025 test subsample, StandardScaler on X, predictors =
xP + era5_features. Loads Optuna-selected hyperparameters from
``Data/Output/XGBoost_best_params.txt`` (written by 2.3), fits that model, then
``shap.TreeExplainer`` on a random subset of ``N_SHAP`` rows from the full test year.

Requires: ``xgboost``, ``shap``. Run ``2.3.XGBoost.py`` first.

**Env overrides:** ``N_SHAP`` (default 200), ``SHAP_SEED``, ``DATA_OUT_DIR``,
``TRAIN_YEAR``, ``TEST_YEAR``, ``XGB_BEST_PARAMS`` (path to best-params table).

**Outputs (tabular only):** ``Data/xai_yHxP.txt`` (plotting in a separate script).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent

TARGET = "yH"
BASE_FEATURE = "xP"
COMBO_LABEL = "yHxP"

# Same predictors as Code/2.3.XGBoost.py / 2.1.MLR.R / 2.4.TabPFN.py
ERA5_FEATURES = ["SZA", "lcc", "mcc", "tcsw", "tcwv"]

N_SHAP = int(os.environ.get("N_SHAP", "200"))
SHAP_SEED = int(os.environ.get("SHAP_SEED", "42"))

DATA_OUT_DIR = Path(os.environ.get("DATA_OUT_DIR", str(PROJECT / "Data")))
XGB_BEST_PARAMS = Path(
    os.environ.get(
        "XGB_BEST_PARAMS",
        str(PROJECT / "Data" / "Output" / "XGBoost_best_params.txt"),
    )
)

TRAIN_YEAR = int(os.environ.get("TRAIN_YEAR", "2024"))
TEST_YEAR = int(os.environ.get("TEST_YEAR", "2025"))

PARAM_COLS = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
    "reg_alpha",
]


def _write_shap_long_txt(
    feature_names: list[str],
    arr: np.ndarray,
    x_shap: pd.DataFrame,
    test_indices: np.ndarray,
    meta_df: pd.DataFrame | None,
    out_path: Path,
) -> None:
    n_s, _ = arr.shape
    rows: list[dict] = []
    for i in range(n_s):
        tpi = int(test_indices[i])
        row_base: dict = {
            "combo": COMBO_LABEL,
            "target": TARGET,
            "sample_index": i,
            "test_year_row_index": tpi,
        }
        if meta_df is not None and len(meta_df) == n_s:
            for col in meta_df.columns:
                row_base[col] = meta_df.iloc[i][col]
        for j, feat in enumerate(feature_names):
            r = dict(row_base)
            r["feature"] = feat
            r["shap_value"] = float(arr[i, j])
            r["feature_value_scaled"] = float(x_shap.iloc[i, j])
            rows.append(r)
    long_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, sep="\t", index=False, float_format="%.8g")
    print(f"Wrote: {out_path}")


def _load_best_params(params_path: Path, combo: str) -> dict[str, Any]:
    if not params_path.is_file():
        raise FileNotFoundError(
            f"Missing {params_path}; run Code/2.3.XGBoost.py first."
        )
    df = pd.read_csv(params_path, sep="\t")
    hit = df.loc[df["combo"].astype(str) == combo]
    if hit.empty:
        raise ValueError(f"No best-params row for combo={combo} in {params_path}")
    row = hit.iloc[0]
    params: dict[str, Any] = {}
    for col in PARAM_COLS:
        if col not in row.index:
            raise ValueError(f"Column {col!r} missing from {params_path}")
        val = row[col]
        if col in ("n_estimators", "max_depth"):
            params[col] = int(val)
        else:
            params[col] = float(val)
    return params


def _train_xgb_best(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> Any:
    import xgboost as xgb

    model = xgb.XGBRegressor(
        random_state=123,
        n_jobs=1,
        objective="reg:squarederror",
        tree_method="hist",
        **params,
    )
    model.fit(X_train, y_train)
    print(f"XGBoost fitted with Optuna best params ({COMBO_LABEL}): {params}")
    return model


def _tree_shap_values(model: Any, x_shap: pd.DataFrame) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    X = x_shap.to_numpy(dtype=float)
    out = explainer(X)
    return np.asarray(out.values, dtype=float)


def main() -> None:
    data_file = PROJECT / "Data" / "arranged15min.txt"
    if not data_file.is_file():
        print(f"ERROR: Missing {data_file}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(data_file, sep="\t")
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)
    df["SZA"] = np.cos(np.radians(df["SZA"]))
    yt = df["Time"].dt.year
    df_train = df.loc[yt == TRAIN_YEAR].copy()
    df_test = df.loc[yt == TEST_YEAR].copy().reset_index(drop=True)

    feature_names: list[str] = [BASE_FEATURE] + ERA5_FEATURES
    missing = [c for c in feature_names if c not in df_train.columns]
    if missing:
        print(f"ERROR: Missing feature columns: {missing}", file=sys.stderr)
        sys.exit(1)

    X_train_raw = df_train[feature_names]
    y_train = df_train[TARGET]
    X_test_raw = df_test[feature_names]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s = scaler.transform(X_test_raw)

    X_test_df = pd.DataFrame(X_test_s, columns=feature_names)

    rng = np.random.default_rng(SHAP_SEED)
    n_full = len(X_test_df)
    if n_full < N_SHAP:
        print(
            f"WARNING: test rows ({n_full}) < N_SHAP ({N_SHAP}); using all.",
            file=sys.stderr,
        )
        n_shap = n_full
        sub = np.arange(n_full, dtype=int)
    else:
        n_shap = N_SHAP
        sub = np.sort(rng.choice(n_full, size=N_SHAP, replace=False))

    x_shap = X_test_df.iloc[sub][feature_names].copy().reset_index(drop=True)
    test_row_indices = sub.astype(int).tolist()

    meta_cols = [c for c in ("Time",) if c in df_test.columns]
    meta_shap = None
    if meta_cols:
        meta_shap = df_test.iloc[test_row_indices][meta_cols].reset_index(drop=True)

    best_params = _load_best_params(XGB_BEST_PARAMS, COMBO_LABEL)
    print(
        f"combo={COMBO_LABEL}  |  train n={X_train_s.shape[0]}  "
        f"|  SHAP rows={n_shap}  |  features={len(feature_names)}  "
        f"|  params={XGB_BEST_PARAMS}"
    )

    model = _train_xgb_best(X_train_s, y_train.values, best_params)
    arr = _tree_shap_values(model, x_shap)

    if arr.ndim != 2 or arr.shape[0] != n_shap:
        print(
            f"WARNING: unexpected SHAP shape {arr.shape}, expected ({n_shap}, n_features)",
            file=sys.stderr,
        )

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx_arr = np.asarray(test_row_indices, dtype=int)
    txt_path = DATA_OUT_DIR / "xai_yHxP.txt"
    _write_shap_long_txt(feature_names, arr, x_shap, idx_arr, meta_shap, txt_path)


if __name__ == "__main__":
    main()
