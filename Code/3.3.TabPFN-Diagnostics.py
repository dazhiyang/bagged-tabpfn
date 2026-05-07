#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import pandas as pd
import sklearn as sk
import torch
from tabpfn import TabPFNRegressor

###############################################################################
# Hard-coded parameters (edit here only)
###############################################################################
PROJECT_PATH = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
INPUT_FILE = os.path.join(PROJECT_PATH, "Data", "arranged15min.txt")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "Data", "Output", "Diag")

TABPFN_FILE = os.path.join(PROJECT_PATH, "Data", "Output", "TabPFN.txt")
TABPFN_C10_FILE = os.path.join(PROJECT_PATH, "Data", "Output", "TabPFN-C10.txt")

TRAIN_YEAR = 2024
TEST_YEAR = 2025
TARGET = "yH"
BASE_FEATURE = "xP"
ERA5_FEATURES = ["SZA", "lcc", "mcc", "tcsw", "tcwv"]
COMBO = "yHxP"

CAP2000_N = 2000
CONTEXT_SAMPLE_SEED = 123
C10_MEMBER_INDEX_1_BASED = 10
COMBO_ORDER = ["yHxP", "yHxS", "yLxP", "yLxS"]
MODEL_RANDOM_STATE = 123
LAYER_INDICES = [3, 6, 9, 12, 15]  # 1-based layer numbers

DEVICE = "auto"
N_ESTIMATORS = 1

PRED_FULL_FILE = os.path.join(OUTPUT_DIR, "pred_full_all_yHxP.csv")
PRED_CAP_FILE = os.path.join(OUTPUT_DIR, "pred_cap2000_all_yHxP.csv")
PRED_BOTH_FILE = os.path.join(OUTPUT_DIR, "pred_context_compare_all_yHxP.csv")

ATTN_FULL_FILE = os.path.join(OUTPUT_DIR, "attention_received_mean_full.csv")
ATTN_CAP_FILE = os.path.join(OUTPUT_DIR, "attention_received_mean_cap2000.csv")
ATTN_DELTA_FILE = os.path.join(OUTPUT_DIR, "attention_delta_cap2000_minus_full.csv")
ATTN_RECEIVED_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "attention_received_summary.csv")
ATTN_LAYER_LONG_FILE = os.path.join(OUTPUT_DIR, "attention_layers_long.csv")

EMBED_PER_TEST_FILE = os.path.join(OUTPUT_DIR, "embedding_per_test.csv")
EMBED_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "embedding_summary.csv")
EMBED_PCA_FILE = os.path.join(OUTPUT_DIR, "embedding_pca_per_test.csv")
EMBED_SPACE_METRIC_FILE = os.path.join(OUTPUT_DIR, "embedding_space_metrics.csv")
METRIC_FILE = os.path.join(OUTPUT_DIR, "metrics_all_yHxP.csv")
BASELINE_COMPARE_FILE = os.path.join(OUTPUT_DIR, "baseline_tabpfn_vs_c10_yHxP.csv")
MANIFEST_FILE = os.path.join(OUTPUT_DIR, "diag_manifest.json")

###############################################################################
# Baseline confirmation from TabPFN.txt vs TabPFN-C10.txt
###############################################################################
os.makedirs(OUTPUT_DIR, exist_ok=True)

tabpfn_ref = pd.read_csv(TABPFN_FILE, sep="\t")
tabpfn_c10 = pd.read_csv(TABPFN_C10_FILE, sep="\t")

baseline_rows = []
for model_name, frame in [("TabPFN", tabpfn_ref), ("TabPFN-C10", tabpfn_c10)]:
    d = frame.loc[frame["combo"] == COMBO].copy()
    rmse = np.sqrt(np.mean((d["x"] - d["y"]) ** 2))
    nrmse = rmse / np.mean(d["y"]) * 100.0
    mbe = np.mean(d["x"] - d["y"])
    baseline_rows.append(
        {
            "combo": COMBO,
            "model": model_name,
            "n": int(len(d)),
            "RMSE": float(rmse),
            "nRMSE_percent": float(nrmse),
            "MBE": float(mbe),
        }
    )

baseline_compare = pd.DataFrame(baseline_rows)
baseline_compare.to_csv(BASELINE_COMPARE_FILE, index=False)

###############################################################################
# Data preparation (same conventions as existing scripts)
###############################################################################
df = pd.read_csv(INPUT_FILE, sep="\t")
df["Time"] = pd.to_datetime(df["Time"], format="mixed")
df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)
df["SZA"] = np.cos(np.radians(df["SZA"]))

year_tag = df["Time"].dt.year
df_train = df.loc[year_tag == TRAIN_YEAR].copy()
df_test_full = df.loc[year_tag == TEST_YEAR].copy()

test_idx = np.arange(len(df_test_full))
df_test = df_test_full.copy()

features = [BASE_FEATURE] + ERA5_FEATURES
token_names = features + ["label"]

X_train = df_train[features]
y_train = df_train[TARGET]
X_test = df_test[features]
y_test = df_test[TARGET]

scaler = sk.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

idx_full = np.arange(X_train_scaled.shape[0])

rng_context = np.random.default_rng(CONTEXT_SAMPLE_SEED)
n_tr = X_train_scaled.shape[0]
idx_cap = None
for k in range(C10_MEMBER_INDEX_1_BASED):
    for combo_name in COMBO_ORDER:
        draw = rng_context.integers(low=0, high=n_tr, size=CAP2000_N)
        if (k == C10_MEMBER_INDEX_1_BASED - 1) and (combo_name == COMBO):
            idx_cap = draw

context_indices = {"full": idx_full, "cap2000": idx_cap}

layer_zero_based = [x - 1 for x in LAYER_INDICES]
layer_name_map = {z: f"L{x}" for z, x in zip(layer_zero_based, LAYER_INDICES)}

###############################################################################
# Attention extractor on selected per-sample feature attention modules
###############################################################################
def register_attention_hooks(model: TabPFNRegressor):
    hooks = []
    layer_chunks = {layer_name_map[i]: [] for i in layer_zero_based}

    for i in layer_zero_based:
        module = model.models_[0].blocks[i].per_sample_attention_between_features
        layer_label = layer_name_map[i]

        def make_hook(label):
            def hook_fn(mod, inputs, _output):
                x = inputs[0].detach()
                br, token_n, _ = x.shape
                q = mod.q_projection(x).view(br, token_n, mod.num_heads, mod.head_dim).permute(0, 2, 1, 3)
                k = mod.k_projection(x).view(br, token_n, mod.num_heads, mod.head_dim).permute(0, 2, 1, 3)
                scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(mod.head_dim)
                weights = torch.softmax(scores, dim=-1)
                layer_chunks[label].append(weights.mean(dim=(0, 1)).detach().cpu().numpy())

            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(layer_label)))

    return hooks, layer_chunks


def fisher_ratio(x: np.ndarray, labels: np.ndarray) -> float:
    classes = np.unique(labels)
    mu = x.mean(axis=0)
    sb = 0.0
    sw = 0.0
    for c in classes:
        xc = x[labels == c]
        if len(xc) == 0:
            continue
        muc = xc.mean(axis=0)
        sb += float(len(xc)) * float(np.sum((muc - mu) ** 2))
        sw += float(np.sum((xc - muc) ** 2))
    return float(sb / (sw + 1e-12))


###############################################################################
# Fit/predict/diagnostics per context
###############################################################################
all_pred_blocks = []
all_embed_blocks = []
attn_by_context_layer = {"full": {}, "cap2000": {}}
metric_rows = []
emb_test_by_context: dict[str, np.ndarray] = {}
y_true_by_context: dict[str, np.ndarray] = {}
abs_err_by_context: dict[str, np.ndarray] = {}

for context_name in ["full", "cap2000"]:
    idx_use = context_indices[context_name]
    X_sub = np.asarray(X_train_scaled[idx_use])
    y_sub = np.asarray(y_train.iloc[idx_use])

    model = TabPFNRegressor(
        n_estimators=N_ESTIMATORS,
        device=DEVICE,
        ignore_pretraining_limits=True,
        random_state=MODEL_RANDOM_STATE,
        fit_mode="fit_preprocessors",
    )
    model.fit(X_sub, y_sub)

    hooks, layer_chunks = register_attention_hooks(model)
    y_pred = np.asarray(model.predict(np.asarray(X_test_scaled)), dtype=float).ravel()
    for h in hooks:
        h.remove()

    for layer_label in layer_chunks:
        attn_by_context_layer[context_name][layer_label] = np.mean(np.stack(layer_chunks[layer_label], axis=0), axis=0)

    emb_train = np.asarray(model.get_embeddings(np.asarray(X_sub), data_source="train"))[0]
    emb_test = np.asarray(model.get_embeddings(np.asarray(X_test_scaled), data_source="test"))[0]

    train_centroid = emb_train.mean(axis=0)
    train_centroid_norm = np.linalg.norm(train_centroid)
    test_norm = np.linalg.norm(emb_test, axis=1)
    cos_to_train = np.dot(emb_test, train_centroid) / (test_norm * train_centroid_norm)

    y_pred_scaled = y_pred * df_test["Ghc"].to_numpy()
    y_true_scaled = y_test.to_numpy() * df_test["Ghc"].to_numpy()
    abs_err = np.abs(y_pred_scaled - y_true_scaled)
    emb_test_by_context[context_name] = emb_test
    y_true_by_context[context_name] = y_true_scaled
    abs_err_by_context[context_name] = abs_err

    pred_block = pd.DataFrame(
        {
            "Time": pd.to_datetime(df_test["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "combo": COMBO,
            "context": context_name,
            "y": np.round(y_true_scaled, 2),
            "x": np.round(y_pred_scaled, 2),
            "abs_error": np.round(abs_err, 6),
        }
    )
    all_pred_blocks.append(pred_block)

    embed_block = pd.DataFrame(
        {
            "Time": pd.to_datetime(df_test["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "combo": COMBO,
            "context": context_name,
            "embedding_norm": test_norm,
            "cos_to_train_centroid": cos_to_train,
            "abs_error": abs_err,
        }
    )
    all_embed_blocks.append(embed_block)

    metric_rows.append(
        {
            "combo": COMBO,
            "context": context_name,
            "n_train_context": int(len(idx_use)),
            "n_test": int(len(df_test)),
            "MBE": float(np.mean(y_pred_scaled - y_true_scaled)),
            "RMSE": float(np.sqrt(np.mean((y_pred_scaled - y_true_scaled) ** 2))),
            "nMBE_percent": float(np.mean(y_pred_scaled - y_true_scaled) / np.mean(y_true_scaled) * 100.0),
            "nRMSE_percent": float(np.sqrt(np.mean((y_pred_scaled - y_true_scaled) ** 2)) / np.mean(y_true_scaled) * 100.0),
        }
    )

###############################################################################
# Write prediction outputs
###############################################################################
pred_all = pd.concat(all_pred_blocks, ignore_index=True)
pred_all.loc[pred_all["context"] == "full"].to_csv(PRED_FULL_FILE, index=False)
pred_all.loc[pred_all["context"] == "cap2000"].to_csv(PRED_CAP_FILE, index=False)
pred_all.to_csv(PRED_BOTH_FILE, index=False)

metric_df = pd.DataFrame(metric_rows)
metric_df.to_csv(METRIC_FILE, index=False)

###############################################################################
# Write layer-specific and aggregate attention outputs
###############################################################################
layer_rows = []
for layer_label in [f"L{x}" for x in LAYER_INDICES]:
    full_mat = attn_by_context_layer["full"][layer_label]
    cap_mat = attn_by_context_layer["cap2000"][layer_label]
    delta_mat = cap_mat - full_mat

    for i_from, from_tok in enumerate(token_names):
        for i_to, to_tok in enumerate(token_names):
            layer_rows.append(
                {
                    "layer": layer_label,
                    "context": "full",
                    "from_token": from_tok,
                    "to_token": to_tok,
                    "attention": float(full_mat[i_from, i_to]),
                }
            )
            layer_rows.append(
                {
                    "layer": layer_label,
                    "context": "cap2000",
                    "from_token": from_tok,
                    "to_token": to_tok,
                    "attention": float(cap_mat[i_from, i_to]),
                }
            )
            layer_rows.append(
                {
                    "layer": layer_label,
                    "context": "delta_cap2000_minus_full",
                    "from_token": from_tok,
                    "to_token": to_tok,
                    "attention": float(delta_mat[i_from, i_to]),
                }
            )

attn_layer_long = pd.DataFrame(layer_rows)
attn_layer_long.to_csv(ATTN_LAYER_LONG_FILE, index=False)

full_avg = np.mean(np.stack([attn_by_context_layer["full"][f"L{x}"] for x in LAYER_INDICES], axis=0), axis=0)
cap_avg = np.mean(np.stack([attn_by_context_layer["cap2000"][f"L{x}"] for x in LAYER_INDICES], axis=0), axis=0)
delta_avg = cap_avg - full_avg

attn_full_df = pd.DataFrame(full_avg, index=token_names, columns=token_names)
attn_cap_df = pd.DataFrame(cap_avg, index=token_names, columns=token_names)
attn_delta_df = pd.DataFrame(delta_avg, index=token_names, columns=token_names)

attn_full_df.to_csv(ATTN_FULL_FILE, index=True, index_label="from_token")
attn_cap_df.to_csv(ATTN_CAP_FILE, index=True, index_label="from_token")
attn_delta_df.to_csv(ATTN_DELTA_FILE, index=True, index_label="from_token")

received_summary = pd.DataFrame(
    {
        "to_token": token_names,
        "received_full_mean": attn_full_df.mean(axis=0).to_numpy(),
        "received_cap2000_mean": attn_cap_df.mean(axis=0).to_numpy(),
        "received_delta_cap2000_minus_full": attn_delta_df.mean(axis=0).to_numpy(),
    }
)
received_summary.to_csv(ATTN_RECEIVED_SUMMARY_FILE, index=False)

###############################################################################
# Write embedding diagnostics
###############################################################################
embed_per_test = pd.concat(all_embed_blocks, ignore_index=True)
embed_per_test.to_csv(EMBED_PER_TEST_FILE, index=False)

embed_summary = (
    embed_per_test.groupby("context", as_index=False)
    .agg(
        embedding_norm_mean=("embedding_norm", "mean"),
        embedding_norm_sd=("embedding_norm", "std"),
        cos_to_train_centroid_mean=("cos_to_train_centroid", "mean"),
        cos_to_train_centroid_sd=("cos_to_train_centroid", "std"),
        abs_error_mean=("abs_error", "mean"),
        abs_error_sd=("abs_error", "std"),
    )
)
embed_summary.to_csv(EMBED_SUMMARY_FILE, index=False)

###############################################################################
# Embedding-space projection and separability metrics
###############################################################################
emb_all = np.vstack([emb_test_by_context["full"], emb_test_by_context["cap2000"]])
pca2 = sk.decomposition.PCA(n_components=2, random_state=MODEL_RANDOM_STATE)
pc_all = pca2.fit_transform(emb_all)
n_test = len(emb_test_by_context["full"])
pc_full = pc_all[:n_test]
pc_cap = pc_all[n_test:]

y_ref = y_true_by_context["full"]
bins = pd.qcut(y_ref, q=3, labels=["low", "mid", "high"], duplicates="drop")
bins_np = np.asarray(bins.astype(str))

embed_pca = pd.concat(
    [
        pd.DataFrame(
            {
                "Time": pd.to_datetime(df_test["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                "context": "full",
                "pc1": pc_full[:, 0],
                "pc2": pc_full[:, 1],
                "y_true": y_true_by_context["full"],
                "abs_error": abs_err_by_context["full"],
                "y_bin": bins_np,
            }
        ),
        pd.DataFrame(
            {
                "Time": pd.to_datetime(df_test["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                "context": "cap2000",
                "pc1": pc_cap[:, 0],
                "pc2": pc_cap[:, 1],
                "y_true": y_true_by_context["cap2000"],
                "abs_error": abs_err_by_context["cap2000"],
                "y_bin": bins_np,
            }
        ),
    ],
    ignore_index=True,
)
embed_pca.to_csv(EMBED_PCA_FILE, index=False)

space_metric_rows = [
    {
        "space": "raw_inputs",
        "context": "full",
        "fisher_ratio": fisher_ratio(X_test_scaled, bins_np),
        "dim": int(X_test_scaled.shape[1]),
        "pca_explained_var_1": np.nan,
        "pca_explained_var_2": np.nan,
    },
    {
        "space": "embeddings",
        "context": "full",
        "fisher_ratio": fisher_ratio(emb_test_by_context["full"], bins_np),
        "dim": int(emb_test_by_context["full"].shape[1]),
        "pca_explained_var_1": float(pca2.explained_variance_ratio_[0]),
        "pca_explained_var_2": float(pca2.explained_variance_ratio_[1]),
    },
    {
        "space": "embeddings",
        "context": "cap2000",
        "fisher_ratio": fisher_ratio(emb_test_by_context["cap2000"], bins_np),
        "dim": int(emb_test_by_context["cap2000"].shape[1]),
        "pca_explained_var_1": float(pca2.explained_variance_ratio_[0]),
        "pca_explained_var_2": float(pca2.explained_variance_ratio_[1]),
    },
]
pd.DataFrame(space_metric_rows).to_csv(EMBED_SPACE_METRIC_FILE, index=False)

###############################################################################
# Manifest
###############################################################################
manifest = {
    "project_path": PROJECT_PATH,
    "input_file": INPUT_FILE,
    "output_dir": OUTPUT_DIR,
    "runtime": "tabpfn_local",
    "train_year": TRAIN_YEAR,
    "test_year": TEST_YEAR,
    "combo": COMBO,
    "target": TARGET,
    "features": features,
    "layer_indices_1_based": LAYER_INDICES,
    "cap2000_n": CAP2000_N,
    "context_sample_seed": CONTEXT_SAMPLE_SEED,
    "c10_member_index_1_based": C10_MEMBER_INDEX_1_BASED,
    "combo_order": COMBO_ORDER,
    "model_random_state": MODEL_RANDOM_STATE,
    "device": DEVICE,
    "n_estimators": N_ESTIMATORS,
    "n_train_full": int(len(idx_full)),
    "n_train_cap2000": int(len(idx_cap)),
    "n_test_used": int(len(df_test)),
    "sampled_test_indices": test_idx.tolist(),
    "sampled_test_timestamps": pd.to_datetime(df_test["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
    "attention_matrix_shape": list(attn_full_df.shape),
    "attention_layer_rows": int(len(attn_layer_long)),
    "embedding_rows": int(len(embed_per_test)),
    "baseline_tabpfn_vs_c10_file": BASELINE_COMPARE_FILE,
    "output_files": [
        PRED_FULL_FILE,
        PRED_CAP_FILE,
        PRED_BOTH_FILE,
        ATTN_FULL_FILE,
        ATTN_CAP_FILE,
        ATTN_DELTA_FILE,
        ATTN_RECEIVED_SUMMARY_FILE,
        ATTN_LAYER_LONG_FILE,
        EMBED_PER_TEST_FILE,
        EMBED_SUMMARY_FILE,
        EMBED_PCA_FILE,
        EMBED_SPACE_METRIC_FILE,
        METRIC_FILE,
        BASELINE_COMPARE_FILE,
    ],
}

with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Wrote diagnostics to: {OUTPUT_DIR}")
print("Baseline comparison:")
print(baseline_compare)
print("Full-test metrics:")
print(metric_df)
