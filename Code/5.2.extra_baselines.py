#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
###############################################################################
# Reviewer-requested supplementary analysis (not part of the 2.* production chain).
# Same data pipeline / split / features / Optuna protocol as Code/2.3.XGBoost.py:
#   TPE, N_TRIALS (default 100), 3-fold CV MSE, seed 123, independent study
#   per (model × combo). Models: XGBoost, LightGBM, CatBoost.
# Writes:
#   Data/Output/extra_baselines_compare.txt   (test RMSE per model × combo)
#   Data/Output/extra_baselines_best_params.txt
#   Data/Output/extra_baselines_compare_wide.txt
#   tex/extra_baselines_compare.tex           (LaTeX RMSE table, 2 d.p.)
# Also appends TabPFN / TabPFN-B from Data/Output/TabPFN.txt and TabPFN-B*.txt
# when those files exist. Set EXTRA_TEX_ONLY=1 to rebuild TSV/TeX from existing
# Optuna results without re-running the search.
# If Data/Output/XGBoost.txt exists, also reports that paper Optuna XGBoost RMSE.
###############################################################################

import os
import warnings

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn as sk
import xgboost as xgb
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

###############################################################################
# paths / Optuna knobs (match 2.3.XGBoost.py)
###############################################################################
project_path = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
N_TRIALS = int(os.environ.get("EXTRA_N_TRIALS", os.environ.get("XGB_N_TRIALS", "100")))
N_CV = 3
OPTUNA_SEED = 123
MODELS = ("XGBoost", "LightGBM", "CatBoost")
# "all" = yHxP/yHxS/yLxP/yLxS; or set EXTRA_COMBO=yHxP (etc.) for one combo.
COMBO = os.environ.get("EXTRA_COMBO", "all").strip()
TEX_ONLY = os.environ.get("EXTRA_TEX_ONLY", "0").strip() in {"1", "true", "True", "yes"}
ENS_K = 10
TEX_MODELS = ("XGBoost", "LightGBM", "CatBoost", "TabPFN", "TabPFN-B")
COMBO_ORDER = ("yHxP", "yHxS", "yLxP", "yLxS")
COMBO_TEX = {
    "yHxP": r"$y_\text{H},\, x_\text{P}$",
    "yHxS": r"$y_\text{H},\, x_\text{S}$",
    "yLxP": r"$y_\text{L},\, x_\text{P}$",
    "yLxS": r"$y_\text{L},\, x_\text{S}$",
}

###############################################################################
# data (match 2.3.XGBoost.py)
###############################################################################
file = os.path.join(project_path, "Data", "arranged15min.txt")
df = pd.read_csv(file, sep="\t")
df["Time"] = pd.to_datetime(df["Time"], format="mixed")
df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)
df["SZA"] = np.cos(np.radians(df["SZA"]))

train_year, test_year = 2024, 2025
yt = df["Time"].dt.year
df_train = df.loc[yt == train_year].copy()
df_test = df.loc[yt == test_year].copy()

_all_jobs = [("yH", "xP"), ("yH", "xS"), ("yL", "xP"), ("yL", "xS")]
if COMBO.lower() == "all":
    combo_jobs = list(_all_jobs)
else:
    combo_jobs = [(t, r) for t, r in _all_jobs if f"{t}{r}" == COMBO]
    if not combo_jobs:
        raise SystemExit(f"Unknown EXTRA_COMBO={COMBO!r}; use yHxP/yHxS/yLxP/yLxS or all")
era5_features = ["SZA", "lcc", "mcc", "tcsw", "tcwv"]

out_dir = os.path.join(project_path, "Data", "Output")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "extra_baselines_compare.txt")
params_file = os.path.join(out_dir, "extra_baselines_best_params.txt")
wide_file = os.path.join(out_dir, "extra_baselines_compare_wide.txt")
tex_file = os.path.join(project_path, "tex", "extra_baselines_compare.tex")
xgb_paper_file = os.path.join(out_dir, "XGBoost.txt")
tabpfn_file = os.path.join(out_dir, "TabPFN.txt")


def suggest_params(model_name: str, trial: optuna.Trial) -> dict:
    if model_name == "XGBoost":
        # Identical ranges to Code/2.3.XGBoost.py
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
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 1500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        }
    raise ValueError(model_name)


def build_model(model_name: str, params: dict):
    if model_name == "XGBoost":
        return xgb.XGBRegressor(
            random_state=OPTUNA_SEED,
            n_jobs=1,
            objective="reg:squarederror",
            tree_method="hist",
            **params,
        )
    if model_name == "LightGBM":
        return lgb.LGBMRegressor(
            random_state=OPTUNA_SEED,
            n_jobs=1,
            verbose=-1,
            **params,
        )
    if model_name == "CatBoost":
        return cb.CatBoostRegressor(
            random_seed=OPTUNA_SEED,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    raise ValueError(model_name)


def cv_mse(model_name: str, params: dict, X: np.ndarray, y: np.ndarray) -> float:
    model = build_model(model_name, params)
    scores = sk.model_selection.cross_val_score(
        model,
        X,
        y,
        cv=N_CV,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    return float(-scores.mean())


def rmse_wm2(y_hat_csi, y_csi, ghc):
    y_hat = y_hat_csi * ghc
    y = y_csi * ghc
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def paper_xgb_rmse_by_combo(path: str) -> dict[str, float]:
    if not os.path.isfile(path):
        return {}
    tab = pd.read_csv(path, sep="\t")
    out = {}
    for combo, g in tab.groupby("combo", sort=False):
        out[str(combo)] = float(
            np.sqrt(np.mean((g["x"].to_numpy() - g["y"].to_numpy()) ** 2))
        )
    return out


def tabpfn_b_rmse_by_combo(directory: str, k: int = ENS_K) -> dict[str, float]:
    member_xs = []
    base = None
    for m in range(1, k + 1):
        path = os.path.join(directory, f"TabPFN-B{m}.txt")
        if not os.path.isfile(path):
            return {}
        d = pd.read_csv(path, sep="\t")
        if base is None:
            base = d[["Time", "combo", "y"]].copy()
        member_xs.append(d["x"].to_numpy(dtype=float))
    base = base.copy()
    base["x"] = np.mean(np.column_stack(member_xs), axis=1)
    out = {}
    for combo, g in base.groupby("combo", sort=False):
        out[str(combo)] = float(
            np.sqrt(np.mean((g["x"].to_numpy() - g["y"].to_numpy()) ** 2))
        )
    return out


def append_tabpfn_rows(res: pd.DataFrame) -> pd.DataFrame:
    """Add / replace TabPFN and TabPFN-B rows from production output files."""
    keep = res[~res["model"].isin(["TabPFN", "TabPFN-B"])].copy()
    extra: list[dict] = []
    tabpfn = paper_xgb_rmse_by_combo(tabpfn_file)
    tabpfn_b = tabpfn_b_rmse_by_combo(out_dir, ENS_K)
    for combo in COMBO_ORDER:
        if combo in tabpfn:
            extra.append(
                {
                    "combo": combo,
                    "model": "TabPFN",
                    "test_rmse_wm2": tabpfn[combo],
                    "cv_mse": np.nan,
                    "n_trials": np.nan,
                    "n_cv": np.nan,
                }
            )
        if combo in tabpfn_b:
            extra.append(
                {
                    "combo": combo,
                    "model": "TabPFN-B",
                    "test_rmse_wm2": tabpfn_b[combo],
                    "cv_mse": np.nan,
                    "n_trials": np.nan,
                    "n_cv": np.nan,
                }
            )
    if not extra:
        return keep
    return pd.concat([keep, pd.DataFrame(extra)], ignore_index=True)


def wide_rmse(res: pd.DataFrame) -> pd.DataFrame:
    wide = res.pivot(index="combo", columns="model", values="test_rmse_wm2")
    cols = [c for c in TEX_MODELS if c in wide.columns]
    return wide.reindex(index=[c for c in COMBO_ORDER if c in wide.index], columns=cols)


def write_latex_table(wide: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = list(wide.columns)
    n = len(cols)
    colspec = "l" + "c" * n
    header = " & ".join(["Combo", *cols]) + r" \\"
    body_lines = []
    for combo in wide.index:
        vals = [float(wide.loc[combo, c]) for c in cols]
        best = min(vals)
        cells = [COMBO_TEX.get(str(combo), str(combo))]
        for v in vals:
            txt = f"{v:.2f}"
            if abs(v - best) < 1e-9:
                txt = r"{\bfseries " + txt + "}"
            cells.append(txt)
        body_lines.append(" & ".join(cells) + r" \\")

    caption = (
        r"Test-year (2025) RMSE (W\,m$^{-2}$) for Optuna-tuned gradient-boosted "
        r"tree baselines (XGBoost, LightGBM, CatBoost; TPE, 100 trials, 3-fold CV) "
        r"versus single-context TabPFN and bagged TabPFN-B ($M=10$, "
        r"$n_{\mathrm{boot}}=2000$). Lowest value in each row is bold."
    )
    lines = [
        r"% Generated by Code/5.2.extra_baselines.py — do not edit by hand.",
        r"% \input{tex/extra_baselines_compare} from your main / SI .tex.",
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tb:extra_baselines}",
        r"\small",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\normalsize",
        r"\end{table}",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def finalize_outputs(res: pd.DataFrame, best_rows: list[dict] | None = None) -> None:
    res = append_tabpfn_rows(res)
    # Drop redundant paper-XGB column from long TSV display path (keep Optuna XGBoost).
    res_out = res[~res["model"].eq("XGBoost_Optuna_2.3")].copy()
    res_out.to_csv(out_file, sep="\t", index=False)
    if best_rows is not None:
        pd.DataFrame(best_rows).to_csv(params_file, sep="\t", index=False)
        print(f"Wrote {params_file}", flush=True)

    wide = wide_rmse(res_out)
    wide.round(2).to_csv(wide_file, sep="\t")
    write_latex_table(wide, tex_file)

    print(f"Wrote {out_file}", flush=True)
    print(f"Wrote {wide_file}", flush=True)
    print(f"Wrote {tex_file}", flush=True)
    print("\nTest RMSE (W/m^2):\n", flush=True)
    print(wide.round(2).to_string(), flush=True)
    best = wide.idxmin(axis=1)
    print("\nLowest RMSE model per combo:", flush=True)
    for combo, model in best.items():
        print(f"  {combo}: {model} ({wide.loc[combo, model]:.2f})", flush=True)


paper_xgb = paper_xgb_rmse_by_combo(xgb_paper_file)

if TEX_ONLY:
    if not os.path.isfile(out_file):
        raise SystemExit(f"EXTRA_TEX_ONLY set but missing {out_file}")
    print(f"Rebuilding tables from {out_file} (no Optuna)", flush=True)
    finalize_outputs(pd.read_csv(out_file, sep="\t"), best_rows=None)
    raise SystemExit(0)

print(
    f"Extra baselines Optuna TPE: models={list(MODELS)}, combos="
    f"{[t + r for t, r in combo_jobs]}, n_trials={N_TRIALS}, "
    f"cv={N_CV}, seed={OPTUNA_SEED} | "
    f"train {train_year} n={len(df_train)}, test {test_year} n={len(df_test)}",
    flush=True,
)
if paper_xgb:
    print(f"Also reporting paper XGBoost from {xgb_paper_file}", flush=True)

rows: list[dict] = []
best_rows: list[dict] = []

for target, retrieval in combo_jobs:
    features = [retrieval] + era5_features
    combo = f"{target}{retrieval}"

    X_train = df_train[features]
    y_train = df_train[target].to_numpy()
    X_test = df_test[features]
    y_test = df_test[target].to_numpy()
    ghc = df_test["Ghc"].to_numpy()

    scaler = sk.preprocessing.StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if combo in paper_xgb:
        rows.append(
            {
                "combo": combo,
                "model": "XGBoost_Optuna_2.3",
                "test_rmse_wm2": paper_xgb[combo],
                "cv_mse": np.nan,
                "n_trials": np.nan,
                "n_cv": np.nan,
            }
        )

    for model_name in MODELS:
        def objective(trial: optuna.Trial, _name=model_name) -> float:
            return cv_mse(_name, suggest_params(_name, trial), X_tr, y_train)

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=OPTUNA_SEED),
            study_name=f"{model_name}_{combo}",
        )
        print(
            f"{combo}: {model_name} Optuna optimize ({N_TRIALS} trials) …",
            flush=True,
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        best_params = dict(study.best_params)
        best_model = build_model(model_name, best_params)
        best_model.fit(X_tr, y_train)
        y_hat = np.asarray(best_model.predict(X_te), dtype=float).ravel()
        rmse = rmse_wm2(y_hat, y_test, ghc)

        rows.append(
            {
                "combo": combo,
                "model": model_name,
                "test_rmse_wm2": rmse,
                "cv_mse": study.best_value,
                "n_trials": N_TRIALS,
                "n_cv": N_CV,
            }
        )
        best_rows.append(
            {
                "combo": combo,
                "model": model_name,
                "cv_mse": study.best_value,
                "test_rmse_wm2": rmse,
                "n_trials": N_TRIALS,
                "n_cv": N_CV,
                **best_params,
            }
        )
        print(
            f"{combo}: {model_name} best cv_mse={study.best_value:.6g}  "
            f"test_RMSE={rmse:.2f} W/m2  params={best_params}",
            flush=True,
        )

finalize_outputs(pd.DataFrame(rows), best_rows=best_rows)
