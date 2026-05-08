#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.4.attention.py

Classifier-checkpoint attention extraction (group_size=1 checkpoint):
- Download/load TabPFN v2 research classifier checkpoint from HuggingFace.
- Quantize regression target for classifier fitting.
- Extract attention for full and B10 contexts.
- Save feature-aggregated attention output used by 4.4.Fig.4.R.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import sklearn as sk
import torch
from tabpfn import TabPFNClassifier

PROJECT_PATH = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
INPUT_FILE = os.path.join(PROJECT_PATH, "Data", "arranged15min.txt")
DIAG_DIR = os.path.join(PROJECT_PATH, "Data", "Output", "Diag")
CHECKPOINT_FILE = os.path.join(PROJECT_PATH, "tabpfn-v2-classifier-gn2p4bpt.ckpt")
CHECKPOINT_URL = "https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/tabpfn-v2-classifier-gn2p4bpt.ckpt"

TRAIN_YEAR = 2024
TEST_YEAR = 2025
TARGET = "yH"
FEATURES = ["xP", "SZA", "lcc", "mcc", "tcsw", "tcwv"]
COMBO = "yHxP"

N_TARGET_BINS = 7
N_ESTIMATORS = 1
LAYERS = [3, 6, 9, 12]
PRED_BATCH_SIZE = 256
TEST_ATTN_SAMPLE_N = 1024
TEST_ATTN_SAMPLE_SEED = 2026

ENS_K = 10
ENS_SEED = 123
BOOTSTRAP_N = 2000
B10_MEMBER_INDEX_1_BASED = 10
COMBO_ORDER = ["yHxP", "yHxS", "yLxP", "yLxS"]

OUT_FEATURE_LONG = os.path.join(DIAG_DIR, "attention_feature_layers_long.csv")


def ensure_checkpoint(local_path: str, url: str) -> None:
    if os.path.exists(local_path):
        return
    urllib.request.urlretrieve(url, local_path)


def quantile_bin_labels(y: pd.Series, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y.to_numpy(dtype=float), q)
    edges = np.unique(edges)
    if len(edges) < 2:
        raise RuntimeError("Failed to build quantile bins: target is nearly constant.")
    yb = pd.cut(y, bins=edges, include_lowest=True, labels=False)
    if yb.isna().any():
        yb = yb.fillna(len(edges) - 2)
    return yb.to_numpy(dtype=int), edges


def build_b10_indices(n_train: int) -> np.ndarray:
    rng = np.random.default_rng(ENS_SEED)
    idx_b10 = None
    for k in range(ENS_K):
        for combo_name in COMBO_ORDER:
            draw = rng.integers(low=0, high=n_train, size=BOOTSTRAP_N)
            if (k == B10_MEMBER_INDEX_1_BASED - 1) and (combo_name == COMBO):
                idx_b10 = draw
    if idx_b10 is None:
        raise RuntimeError("Failed to derive B10 sampling indices.")
    return idx_b10


def feature_and_label_blocks(token_n: int, n_features: int) -> tuple[list[str], list[list[int]]]:
    # Use all non-last tokens for feature blocks and reserve the last token as label.
    # Feature tokens are split contiguously into n_features blocks.
    if token_n < 2:
        return FEATURES + ["label"], [[] for _ in FEATURES] + [[0]]
    feat_idx = np.arange(token_n - 1)
    feat_blocks = [list(x) for x in np.array_split(feat_idx, n_features)]
    labels = FEATURES + ["label"]
    blocks = feat_blocks + [[token_n - 1]]
    return labels, blocks


def extract_attention(model: TabPFNClassifier, x_test: np.ndarray, context: str, layers_1_based: list[int]) -> list[dict]:
    feature_rows: list[dict] = []

    layer_buffers: dict[int, list[np.ndarray]] = {l: [] for l in layers_1_based}
    hooks = []
    for l1 in layers_1_based:
        module = model.models_[0].transformer_encoder.layers[l1 - 1].self_attn_between_features

        def make_hook(layer_id: int):
            def hook_fn(mod, inputs, _output):
                x = inputs[0].detach()
                q, k, _v, kv, qkv = mod.compute_qkv(
                    x=x,
                    x_kv=None,
                    k_cache=mod._k_cache,
                    v_cache=mod._v_cache,
                    kv_cache=mod._kv_cache,
                    cache_kv=False,
                    use_cached_kv=False,
                    reuse_first_head_kv=False,
                )
                if qkv is not None:
                    q, k, _ = qkv.unbind(dim=-3)
                elif kv is not None and q is not None:
                    k, _ = kv.unbind(dim=-3)
                elif q is None or k is None:
                    return
                d_k = q.shape[-1]
                logits = torch.einsum("bsthd,bskhd->bstkh", q, k) / np.sqrt(float(d_k))
                attn = torch.softmax(logits, dim=3)
                layer_buffers[layer_id].append(attn.mean(dim=(0, 1, 4)).detach().cpu().numpy())

            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(l1)))

    for i in range(0, len(x_test), PRED_BATCH_SIZE):
        _ = model.predict(x_test[i : i + PRED_BATCH_SIZE])
    for h in hooks:
        h.remove()

    for l1 in layers_1_based:
        mats = layer_buffers[l1]
        if len(mats) == 0:
            continue
        mat = np.mean(np.stack(mats, axis=0), axis=0)
        token_n = mat.shape[0]

        feat_names, blocks = feature_and_label_blocks(token_n=token_n, n_features=len(FEATURES))
        for i, bi in enumerate(blocks):
            for j, bj in enumerate(blocks):
                if len(bi) == 0 or len(bj) == 0:
                    attn_val = np.nan
                else:
                    attn_val = float(np.mean(mat[np.ix_(bi, bj)]))
                feature_rows.append(
                    {
                        "context": context,
                        "layer": f"L{l1}",
                        "from_feature": feat_names[i],
                        "to_feature": feat_names[j],
                        "attention": float(attn_val),
                    }
                )

    return feature_rows


os.makedirs(DIAG_DIR, exist_ok=True)
ensure_checkpoint(CHECKPOINT_FILE, CHECKPOINT_URL)

df = pd.read_csv(INPUT_FILE, sep="\t")
df["Time"] = pd.to_datetime(df["Time"], format="mixed")
df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)
df["SZA"] = np.cos(np.radians(df["SZA"]))

yt = df["Time"].dt.year
df_train = df.loc[yt == TRAIN_YEAR].copy()
df_test = df.loc[yt == TEST_YEAR].copy()

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_test = df_test[FEATURES]

scaler = sk.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rng_test = np.random.default_rng(TEST_ATTN_SAMPLE_SEED)
n_test_total = X_test_scaled.shape[0]
if TEST_ATTN_SAMPLE_N < n_test_total:
    idx_test_attn = np.sort(rng_test.choice(n_test_total, size=TEST_ATTN_SAMPLE_N, replace=False))
else:
    idx_test_attn = np.arange(n_test_total)
X_test_attn = np.asarray(X_test_scaled[idx_test_attn])

y_train_bin, _ = quantile_bin_labels(y_train, N_TARGET_BINS)
idx_b10 = build_b10_indices(n_train=X_train_scaled.shape[0])

model_full = TabPFNClassifier(
    model_path=CHECKPOINT_FILE,
    n_estimators=N_ESTIMATORS,
    ignore_pretraining_limits=True,
    random_state=ENS_SEED,
)
model_full.fit(X_train_scaled, y_train_bin)

model_b10 = TabPFNClassifier(
    model_path=CHECKPOINT_FILE,
    n_estimators=N_ESTIMATORS,
    ignore_pretraining_limits=True,
    random_state=int(ENS_SEED + (B10_MEMBER_INDEX_1_BASED - 1)),
)
model_b10.fit(np.asarray(X_train_scaled[idx_b10]), np.asarray(y_train_bin[idx_b10]))

n_layers = len(model_full.models_[0].transformer_encoder.layers)
layers_used = [l for l in LAYERS if l <= n_layers]

feature_rows = extract_attention(model_full, X_test_attn, "full", layers_used)
feature_rows.extend(extract_attention(model_b10, X_test_attn, "b10", layers_used))
pd.DataFrame(feature_rows).to_csv(OUT_FEATURE_LONG, index=False)

print("Classifier checkpoint attention extraction complete.")
print(f"Wrote: {OUT_FEATURE_LONG}")
