# 📊 Site_Adaptation

📖 Research code for site adaptation experiments with retrieval-, regression-, and TabPFN-based models, manuscript figures, and TabPFN attention / embedding diagnostics.

## 👥 Authors

- 👤 **Dazhi Yang** · 🏫 School of Electrical Engineering and Automation, Harbin Institute of Technology (HIT) · 📧 yangdazhi.nus@gmail.com  
- 👤 **Yun Chen** (_PowerPuffYun_) · 🏛️ Public Meteorological Service Center, China Meteorological Administration (CMA) · 📧 chenyunpku@163.com  

## 📂 Repository layout

| Path | Role |
|------|------|
| `Code/` | All runnable scripts (numbered by stage); see **Scripts in `Code/`** below. |
| `Data/arranged15min.txt` | Main arranged 15 min site dataset (tracked). Raw outputs go to `Data/Output/` (gitignored). |
| `Data/Output/Diag/` | TabPFN diagnostics CSVs / PNGs (gitignored unless you add them): attention (`attention_feature_layers_long.csv`), PCA long (`feature_token_pca_layers_long.csv`), etc. |
| `tex/` | LaTeX fragments (e.g. `baseline_tables.tex`), manuscript PDFs such as `Fig1.pdf`–`Fig3.pdf` when built by `4.*` scripts. |

## 🧠 TabPFN v2 classifier checkpoint

Local **TabPFN v2 classifier** weights file (project root):

**`tabpfn-v2-classifier-gn2p4bpt.ckpt`**

Used by **`Code/3.3.attention.py`**, **`Code/3.4.embedding.py`**, and **`Code/3.10.mininalExample.py`** (`CHECKPOINT_FILE` / `model_path`). Place this file at the repo root or edit each script’s path. These scripts use the **pip `tabpfn`** stack (not `tabpfn_client`).

---

## 📋 Scripts in `Code/`

### `1.*` — Data & interpretability inputs

| File | What it does |
|------|----------------|
| **`1.1.arrange_data_15min.R`** | End-to-end arrangement of raw inputs (NCDF, SolarData, etc.) into **`Data/arranged15min.txt`** (15 min aggregates, solar geometry, covariates). Paths include external raw-data dirs—edit `dir.data` / `dir0` for your machine. |
| **`1.2.XAI.py`** | **SHAP** (`TreeExplainer`) for **XGBoost**, same splits/features as `2.3.XGBoost.py`; writes **`Data/xai_yHxP.txt`** (configurable via `DATA_OUT_DIR`) for **`4.1.Fig.1.R`**. |

### `2.*` — Baselines & cloud TabPFN predictions

| File | What it does |
|------|----------------|
| **`2.0.raw.R`** | **Raw retrieval vs observation** (no bias correction); test-year rows only → **`Data/Output/raw.txt`** (`Time`, `combo`, `y`, `x`, W·m⁻²). |
| **`2.1.MLR.R`** | **Multiple linear regression** (clear-sky index workflow + ERA5 covariates) → **`Data/Output/MLR.txt`**. |
| **`2.2.KCDE.R`** | **KCDE** density-regression baseline → **`Data/Output/KCDE.txt`**. |
| **`2.3.XGBoost.py`** | **XGBoost** + `GridSearchCV`; same predictors as MLR pipeline → **`Data/Output/XGBoost.txt`**. |
| **`2.4.TabPFN.py`** | **TabPFN regressor** via **`tabpfn_client`** (API / cloud defaults, e.g. v2.5) → **`Data/Output/TabPFN.txt`**. |
| **`2.5.TabPFN-B.py`** | **Bagged TabPFN-B**: ensemble members with bootstrap draws → **`Data/Output/TabPFN-B{m}.txt`** for each member `m`. |

### `3.*` — Metrics & TabPFN–Fig. 3 diagnostics

| File | What it does |
|------|----------------|
| **`3.1.overall_metric.R`** | Aggregates method outputs + TabPFN-B ensemble files; writes **`tex/baseline_tables.tex`** (RMSE/MBE tables used by Fig. 3 panel **(a)**). Expects `ENS_K` aligned with `2.5.TabPFN-B.py`. |
| **`3.2.MW-Stats.R`** | **Murphy–Winkler** decomposition statistics from model output files → summary tables / diagnostics (see script headers for targets). |
| **`3.3.attention.py`** | **Attention heatmaps data**: hooks `self_attn_between_features`, chunked `predict`, **full vs B10** contexts → **`Data/Output/Diag/attention_feature_layers_long.csv`**. Uses **`tabpfn-v2-classifier-gn2p4bpt.ckpt`**. |
| **`3.4.embedding.py`** | **Token embeddings + PCA** for Fig. 3 panel **(c)** (Full vs B10, staged Input/L layers) → **`Data/Output/Diag/feature_token_pca_layers_long.csv`**. Uses **`tabpfn-v2-classifier-gn2p4bpt.ckpt`**. Run **`3.3.attention.py`** separately for panels **(b)**–**(c)** attention part. |
| **`3.10.mininalExample.py`** | Standalone **attribute-token PCA / diagnostic PNGs** (minimal TabPFN hook demo). Uses **`tabpfn-v2-classifier-gn2p4bpt.ckpt`**; not required for main `4.3.Fig.3.R` if you already run `3.4`. |

### `4.*` — Manuscript figures

| File | What it does |
|------|----------------|
| **`4.1.Fig.1.R`** | **Fig. 1** PDF: scatter + marginals + SHAP beeswarm (consumes XAI output from `1.2.XAI.py`). |
| **`4.2.Fig.2.R`** | **Fig. 2** ensemble / diagnostics composite (see script title: ensemble diagnostics). |
| **`4.3.Fig.3.R`** | **Fig. 3** PDF (`tex/Fig3.pdf`): **(a)** ΔRMSE vs TabPFN from **`tex/baseline_tables.tex`**; **(b)** attention + B10−Full delta from **`attention_feature_layers_long.csv`**; **(c)** PCA scatter from **`feature_token_pca_layers_long.csv`** (needs **`ggh4x`**). |

## 📦 Prerequisites

- **R** — tidyverse-style packages as required by each script (e.g. `ggplot2`, `dplyr`, `patchwork`; Fig. 3 also needs `ggnewscale`, **`ggh4x`**, `scattermore`).
- **Python** — Anaconda-style env recommended; packages include `numpy`, `pandas`, `scikit-learn`, `torch`, **`tabpfn`**, and for embedding diagnostics **`tabpfn_extensions`**. TabPFN client scripts use **`tabpfn_client`**.

## 🗂️ Paths and checkpoints

Most scripts use a **hardcoded `project_path` / `dir0`** pointing at the original machine layout. After cloning, search and replace that path (or refactor to a single config) so `Data/`, `tex/`, and **`tabpfn-v2-classifier-gn2p4bpt.ckpt`** resolve on your system.

## 🚀 Illustrative pipeline (not exhaustive)

1. **`Code/1.1.arrange_data_15min.R`** — Build / maintain `Data/arranged15min.txt`.
2. **`Code/2.*`** — Baselines (e.g. MLR, KCDE, XGBoost), raw retrieval export, TabPFN / TabPFN-B client predictions; outputs under `Data/Output/`.
3. **`Code/3.*`** — Overall metrics, Murphy–Winkler stats, **TabPFN attention** (`3.3.attention.py`) and **token PCA / embeddings** (`3.4.embedding.py`) for Fig. 3; optional minimal embedding example (`3.10.mininalExample.py`).
4. **`Code/4.*`** — Assemble **Fig. 1–3** PDFs in `tex/` (e.g. run `4.3.Fig.3.R` after the Diag CSVs exist).

### 🔧 Environment variables (optional)

- **`3.3.attention.py`**: `TABPFN_PREDICT_DEVICE`, `TABPFN_DEVICE` (default in script: `mps`); `TABPFN_ATTENTION_SKIP_IF_EXISTS=1` skips a rerun if the attention CSV already exists.
- **`3.4.embedding.py`**: `TABPFN_EMBED_DEVICE` (default `cpu`).

## ⚖️ License

[MIT](LICENSE) (Copyright 2026 Dazhi Yang — see file for full text).
