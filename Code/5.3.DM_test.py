#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
###############################################################################
# Reviewer-requested supplementary analysis (5.* series).
# Diebold-Mariano (1995) tests on paired 15-min predictions (2025 test year).
#
# Builds a 6 x 6 win-count matrix:
#   rows    = model B
#   columns = model A
#   cell    = number of combos (out of 4) where A significantly beats B
#             (two-sided DM, alpha = ALPHA; lower mean squared error for A).
#
# Models: Raw, MLR, KCDE, XGBoost, TabPFN, TabPFN-B (bagged mean of B1…B_K).
#
# Writes:
#   Data/Output/dm_win_matrix.txt          wide matrix (for plotting)
#   Data/Output/dm_test_detail.txt         per combo x pair (stat, p, win)
#   Data/Output/dm_win_matrix_long.txt     long format (model_a, model_b, n_wins)
###############################################################################

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

###############################################################################
# paths / parameters
###############################################################################
project_path = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
out_dir = os.path.join(project_path, "Data", "Output")

COMBO_ORDER = ("yHxP", "yHxS", "yLxP", "yLxS")
# Must match ENS_K in Code/2.5.TabPFN-B.py.
ENS_K = 10
MODEL_ORDER = ("Raw", "MLR", "KCDE", "XGBoost", "TabPFN", "TabPFN-B")
MODEL_FILES = {
    "Raw": "raw.txt",
    "MLR": "MLR.txt",
    "KCDE": "KCDE.txt",
    "XGBoost": "XGBoost.txt",
    "TabPFN": "TabPFN.txt",
}

ALPHA = float(os.environ.get("DM_ALPHA", "0.05"))
# HAC lag for 15-min series: default one diurnal cycle (96 x 15-min).
HAC_LAG = int(os.environ.get("DM_HAC_LAG", "96"))
HARVEY_CORR = os.environ.get("DM_HARVEY", "0").strip() in {"1", "true", "True", "yes"}


def read_pred(path: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df.assign(
        Time=pd.to_datetime(df["Time"], format="mixed"),
        model=model,
        y=df["y"].astype(float),
        x=df["x"].astype(float),
    )[["Time", "combo", "y", "x", "model"]]


def newey_west_var_mean(x: np.ndarray, lag: int) -> float:
    """Variance of the sample mean of x with Bartlett-kernel HAC (Newey-West)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return np.nan
    lag = max(0, min(int(lag), n - 1))
    xd = x - x.mean()
    gamma0 = np.dot(xd, xd) / n
    if lag == 0:
        return gamma0 / n
    acc = 0.0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        cov = np.dot(xd[k:], xd[:-k]) / n
        acc += w * cov
    return (gamma0 + 2.0 * acc) / n


def dm_test(
    actual: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    h: int = 1,
    lag: int | None = None,
    harvey: bool = False,
) -> tuple[float, float, float]:
    """
    Diebold-Mariano test (squared-error loss).
    Returns (dm_stat, p_value_two_sided, mean_loss_diff).
    mean_loss_diff = mean(e_a^2 - e_b^2); negative => A better than B.
    """
    y = np.asarray(actual, dtype=float)
    ea = y - np.asarray(pred_a, dtype=float)
    eb = y - np.asarray(pred_b, dtype=float)
    d = ea**2 - eb**2
    n = d.size
    if n < 3:
        return np.nan, np.nan, np.nan

    d_bar = float(d.mean())
    use_lag = HAC_LAG if lag is None else lag
    var_bar = newey_west_var_mean(d, use_lag)
    if not np.isfinite(var_bar) or var_bar <= 0:
        return np.nan, np.nan, d_bar

    dm = d_bar / np.sqrt(var_bar)
    if harvey:
        # Harvey, Leybourne & Newbold (1997) small-sample correction.
        adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm = dm * adj

    p = float(2.0 * (1.0 - stats.norm.cdf(abs(dm))))
    return float(dm), p, d_bar


def load_tabpfn_b_preds(k: int = ENS_K) -> pd.DataFrame:
    """Row-wise mean of TabPFN-B1 … TabPFN-B{k} (same as Code/5.2.extra_baselines.py)."""
    member_xs: list[np.ndarray] = []
    base: pd.DataFrame | None = None
    for m in range(1, k + 1):
        path = os.path.join(out_dir, f"TabPFN-B{m}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        d = pd.read_csv(path, sep="\t")
        if base is None:
            base = d[["Time", "combo", "y"]].copy()
        member_xs.append(d["x"].to_numpy(dtype=float))
    assert base is not None
    out = base.copy()
    out["x"] = np.mean(np.column_stack(member_xs), axis=1)
    out["Time"] = pd.to_datetime(out["Time"], format="mixed")
    out["model"] = "TabPFN-B"
    return out[["Time", "combo", "y", "x", "model"]]


def load_all_preds() -> pd.DataFrame:
    frames = []
    for model, fname in MODEL_FILES.items():
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        frames.append(read_pred(path, model))
    frames.append(load_tabpfn_b_preds())
    out = pd.concat(frames, ignore_index=True)
    return out.loc[out["combo"].isin(COMBO_ORDER)].copy()


def main() -> None:
    all_pred = load_all_preds()
    detail_rows: list[dict] = []

    for combo in COMBO_ORDER:
        sub = (
            all_pred.loc[all_pred["combo"] == combo]
            .sort_values("Time")
            .reset_index(drop=True)
        )
        y = sub.groupby("Time", sort=False)["y"].first()
        wide = sub.pivot(index="Time", columns="model", values="x").reindex(
            columns=MODEL_ORDER
        )
        y_arr = y.reindex(wide.index).to_numpy()
        for model_a in MODEL_ORDER:
            for model_b in MODEL_ORDER:
                if model_a == model_b:
                    continue
                pa = wide[model_a].to_numpy()
                pb = wide[model_b].to_numpy()
                dm_stat, p_val, d_bar = dm_test(y_arr, pa, pb, harvey=HARVEY_CORR)
                a_wins = bool(
                    np.isfinite(p_val)
                    and p_val < ALPHA
                    and np.isfinite(d_bar)
                    and d_bar < 0
                )
                detail_rows.append(
                    {
                        "combo": combo,
                        "model_a": model_a,
                        "model_b": model_b,
                        "dm_stat": dm_stat,
                        "p_value": p_val,
                        "mean_loss_diff_wm2_sq": d_bar,
                        "a_wins": int(a_wins),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    detail_path = os.path.join(out_dir, "dm_test_detail.txt")
    detail.to_csv(detail_path, sep="\t", index=False)

    # Win matrix: row = B, col = A, value = sum of a_wins over combos.
    win_long = (
        detail.groupby(["model_a", "model_b"], as_index=False)["a_wins"]
        .sum()
        .rename(columns={"a_wins": "n_wins"})
    )
    win_mat = win_long.pivot(index="model_b", columns="model_a", values="n_wins")
    win_mat = win_mat.reindex(index=MODEL_ORDER, columns=MODEL_ORDER)
    for m in MODEL_ORDER:
        if m in win_mat.index and m in win_mat.columns:
            win_mat.loc[m, m] = 0

    win_wide_path = os.path.join(out_dir, "dm_win_matrix.txt")
    win_mat.to_csv(win_wide_path, sep="\t")

    win_long_path = os.path.join(out_dir, "dm_win_matrix_long.txt")
    win_long.to_csv(win_long_path, sep="\t", index=False)

    print(
        f"DM tests: models={list(MODEL_ORDER)}, combos={list(COMBO_ORDER)}, "
        f"alpha={ALPHA}, HAC_lag={HAC_LAG}, harvey={HARVEY_CORR}",
        flush=True,
    )
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {win_wide_path}", flush=True)
    print(f"Wrote {win_long_path}", flush=True)
    print(
        "\nWin-count matrix (row=B, col=A; cell = # combos A beats B at p<alpha):\n",
        flush=True,
    )
    print(win_mat.to_string(), flush=True)


if __name__ == "__main__":
    main()
