#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1.2 XAI — SHAP (TreeExplainer) results for **XGBoost**, combo **yHxS** (no figures).

Same statistical setup as Code/2.3.XGBoost.py: clear-sky indices, cos(SZA),
train 2024 / explain on 2025 test subsample, StandardScaler on X. Fits
``GridSearchCV`` with the same grid as 2.3, then ``shap.TreeExplainer`` on a random
subset of ``N_SHAP`` rows from the **full** test year (no cap on test pool size).

Requires: ``xgboost``, ``shap``.

**Env overrides:** ``N_SHAP`` (default 80), ``SHAP_SEED``, ``DATA_OUT_DIR``,
``TRAIN_YEAR``, ``TEST_YEAR``.

**Outputs (tabular only):** ``Data/xai_yHxP.txt`` (plotting in a separate script).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent

TARGET = "yH"
BASE_FEATURE = "xP"
COMBO_LABEL = "yHxP"

N_SHAP = int(os.environ.get("N_SHAP", "200"))
SHAP_SEED = int(os.environ.get("SHAP_SEED", "42"))

DATA_OUT_DIR = Path(os.environ.get("DATA_OUT_DIR", str(PROJECT / "Data")))

TRAIN_YEAR = int(os.environ.get("TRAIN_YEAR", "2024"))
TEST_YEAR = int(os.environ.get("TEST_YEAR", "2025"))

# Match Code/2.3.XGBoost.py
XGB_PARAM_GRID = {
    "n_estimators": [200, 450, 800, 1200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.02, 0.05, 0.12, 0.25],
}


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


def _train_xgb_grid(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    import xgboost as xgb

    base = xgb.XGBRegressor(
        random_state=123,
        n_jobs=1,
        objective="reg:squarederror",
        tree_method="hist",
    )
    grid = sk.model_selection.GridSearchCV(
        base,
        XGB_PARAM_GRID,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"XGBoost GridSearchCV best_params_: {grid.best_params_}")
    return grid.best_estimator_


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

    other_cols = [
        col
        for col in df_train.columns
        if col not in ["Time", "yH", "yL", "xS", "xP", "Ghc"]
    ]
    feature_names: list[str] = [BASE_FEATURE] + other_cols

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

    print(
        f"combo={COMBO_LABEL}  |  train n={X_train_s.shape[0]}  "
        f"|  SHAP rows={n_shap}  |  features={len(feature_names)}"
    )

    model = _train_xgb_grid(X_train_s, y_train.values)
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
