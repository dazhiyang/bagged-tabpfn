#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.5.attention.py — Attention diagnostics only (Fig. 3 panels (b)–(c) CSV).

``predict`` + forward hooks on ``self_attn_between_features`` at ``LAYERS``;
writes ``Data/Output/Diag/attention_feature_layers_long.csv``.

Query pools match ``3.3.bagged.py``: Full = all train rows; B10 = all B10 bootstrap fit rows.
Chunked ``predict`` avoids MPS/GPU OOM on large query matrices.

Run ``3.3.bagged.py`` for ``feature_token_pca_layers_long.csv`` (embeddings + PCA rows).

Device: ``TABPFN_PREDICT_DEVICE`` (default ``auto``), or ``TABPFN_DEVICE`` if set.
"""

import os

import numpy as np
import pandas as pd
import sklearn as sk
import torch
from tabpfn import TabPFNClassifier

PROJECT_PATH = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
INPUT_FILE = os.path.join(PROJECT_PATH, "Data", "arranged15min.txt")
DIAG_DIR = os.path.join(PROJECT_PATH, "Data", "Output", "Diag")
CHECKPOINT_FILE = os.path.join(PROJECT_PATH, "tabpfn-v2-classifier-gn2p4bpt.ckpt")
OUT_FEATURE_LONG = os.path.join(DIAG_DIR, "attention_feature_layers_long.csv")

TRAIN_YEAR = 2024
TARGET = "yH"
FEATURES = ["xP", "SZA", "lcc", "mcc", "tcsw", "tcwv"]
COMBO = "yHxP"

N_TARGET_BINS = 7
N_ESTIMATORS = 1
LAYERS = [1, 2, 3, 6, 9, 12]
ATTENTION_PREDICT_CHUNK = 128

ENS_K = 10
ENS_SEED = 123
BOOTSTRAP_N = 2000
B10_MEMBER_INDEX_1_BASED = 10
COMBO_ORDER = ["yHxP", "yHxS", "yLxP", "yLxS"]

TABPFN_PREDICT_DEVICE = os.environ.get(
    "TABPFN_PREDICT_DEVICE",
    os.environ.get("TABPFN_DEVICE", "auto"),
)


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
    if token_n < 2:
        return FEATURES + ["label"], [[] for _ in FEATURES] + [[0]]
    feat_idx = np.arange(token_n - 1)
    feat_blocks = [list(x) for x in np.array_split(feat_idx, n_features)]
    labels = FEATURES + ["label"]
    blocks = feat_blocks + [[token_n - 1]]
    return labels, blocks


def extract_attention(
    model: TabPFNClassifier,
    x_query: np.ndarray,
    context: str,
    layers_1_based: list[int],
) -> list[dict]:
    feature_rows: list[dict] = []
    # Per layer: list of (batch_size_this_forward, attn_matrix token×token)
    layer_buffers: dict[int, list[tuple[int, np.ndarray]]] = {l: [] for l in layers_1_based}
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
                bsz = int(attn.shape[0])
                mat = attn.mean(dim=(0, 1, 4)).detach().cpu().numpy()
                layer_buffers[layer_id].append((bsz, mat))

            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(l1)))

    n = len(x_query)
    for start in range(0, n, ATTENTION_PREDICT_CHUNK):
        sl = x_query[start : start + ATTENTION_PREDICT_CHUNK]
        _ = model.predict(sl)

    for h in hooks:
        h.remove()

    for l1 in layers_1_based:
        chunks = layer_buffers[l1]
        if len(chunks) == 0:
            continue
        total_w = sum(w for w, _ in chunks)
        mat = sum(w * m for w, m in chunks) / float(total_w)
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


def main() -> None:
    if not os.path.isfile(CHECKPOINT_FILE):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}\n"
            "Place the TabPFN classifier checkpoint at this path or set CHECKPOINT_FILE."
        )

    os.makedirs(DIAG_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, sep="\t")
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)
    df["SZA"] = np.cos(np.radians(df["SZA"]))

    yt = df["Time"].dt.year
    df_train = df.loc[yt == TRAIN_YEAR].copy()

    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]

    scaler = sk.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    y_train_bin, _ = quantile_bin_labels(y_train, N_TARGET_BINS)

    idx_b10 = build_b10_indices(n_train=X_train_scaled.shape[0])

    X_full_attn = np.asarray(X_train_scaled)
    X_b10_attn = np.asarray(X_train_scaled[idx_b10])

    model_full = TabPFNClassifier(
        model_path=CHECKPOINT_FILE,
        n_estimators=N_ESTIMATORS,
        ignore_pretraining_limits=True,
        random_state=ENS_SEED,
        device=TABPFN_PREDICT_DEVICE,
    )
    model_full.fit(X_train_scaled, y_train_bin)

    model_b10 = TabPFNClassifier(
        model_path=CHECKPOINT_FILE,
        n_estimators=N_ESTIMATORS,
        ignore_pretraining_limits=True,
        random_state=int(ENS_SEED + (B10_MEMBER_INDEX_1_BASED - 1)),
        device=TABPFN_PREDICT_DEVICE,
    )
    model_b10.fit(np.asarray(X_train_scaled[idx_b10]), np.asarray(y_train_bin[idx_b10]))

    n_layers = len(model_full.models_[0].transformer_encoder.layers)
    layers_used = [l for l in LAYERS if l <= n_layers]

    feature_rows_full = extract_attention(model_full, X_full_attn, "full", layers_used)
    feature_rows_b10 = extract_attention(model_b10, X_b10_attn, "b10", layers_used)
    pd.DataFrame(feature_rows_full + feature_rows_b10).to_csv(OUT_FEATURE_LONG, index=False)

    print("Attention extraction complete (chunked predict).")
    print(f"Wrote: {OUT_FEATURE_LONG}")
    print("For PCA rows run Code/3.3.bagged.py; for Fig. 3 run Code/4.3.Fig.3.R.")


if __name__ == "__main__":
    main()
