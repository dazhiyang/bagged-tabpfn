#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.4.attention.py

Classifier-checkpoint attention extraction (group_size=1 checkpoint):
- Download/load TabPFN v2 research classifier checkpoint from HuggingFace.
- Quantize regression target for classifier fitting.
- Extract attention for full and B10 contexts.
- Write CSV diagnostics for Code/4.4.Fig.4.R:
  attention_feature_layers_long.csv, feature_token_pca_layers_long.csv,
  raw_attribute_token_map.csv.
No figures here: panel (c) PCA (and panels (a)–(b)) are drawn in 4.4.Fig.4.R.
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
OUT_FEATURE_TOKEN_PCA_LONG = os.path.join(DIAG_DIR, "feature_token_pca_layers_long.csv")
OUT_RAW_TOKEN_MAP = os.path.join(DIAG_DIR, "raw_attribute_token_map.csv")


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


def apply_bin_edges(y: pd.Series, edges: np.ndarray) -> np.ndarray:
    yb = pd.cut(y, bins=edges, include_lowest=True, labels=False)
    if yb.isna().any():
        yb = yb.fillna(len(edges) - 2)
    return yb.to_numpy(dtype=int)


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


def raw_attribute_token_indices(model: TabPFNClassifier, feature_names: list[str]) -> dict[str, int]:
    """Return transformer token indices for raw input attributes after TabPFN preprocessing."""
    if not hasattr(model, "executor_") or not getattr(model.executor_, "ensemble_members", None):
        raise RuntimeError("Model must be fitted before raw attribute token indices can be inspected.")
    if len(model.executor_.ensemble_members) != 1:
        raise RuntimeError("Raw token mapping is only implemented for n_estimators=1.")

    features_per_group = getattr(model.models_[0], "features_per_group", None)
    if features_per_group != 1:
        raise RuntimeError(f"Expected features_per_group=1, got {features_per_group}.")

    member = model.executor_.ensemble_members[0]
    steps = getattr(member.cpu_preprocessor, "steps", [])
    raw_source_indices = list(range(len(feature_names)))
    raw_pre_shuffle: dict[str, int] | None = None
    shuffle_permutation: list[int] | None = None

    for step in steps:
        step_name = step.__class__.__name__
        if step_name == "RemoveConstantFeaturesStep":
            sel = getattr(step, "sel_", None)
            if sel is not None:
                raw_source_indices = [src for src, keep in zip(raw_source_indices, sel) if bool(keep)]
        elif step_name == "ReshapeFeatureDistributionsStep":
            if getattr(step, "append_to_original", False) is not True:
                raise RuntimeError(
                    "TabPFN preprocessing did not preserve raw original features; "
                    f"append_to_original={getattr(step, 'append_to_original', None)!r}."
                )
            # With append_to_original=True, the raw passthrough columns are first.
            raw_pre_shuffle = {
                feature_names[src]: new_idx
                for new_idx, src in enumerate(raw_source_indices)
                if src < len(feature_names)
            }
        elif step_name == "ShuffleFeaturesStep":
            perm = getattr(step, "index_permutation_", None)
            if perm is not None:
                shuffle_permutation = [int(x) for x in perm]

    if raw_pre_shuffle is None:
        raw_pre_shuffle = {name: idx for idx, name in enumerate(feature_names)}

    token_indices: dict[str, int] = {}
    for name in feature_names:
        if name not in raw_pre_shuffle:
            raise RuntimeError(f"Raw feature {name!r} was removed before tokenization.")
        pre_shuffle_idx = raw_pre_shuffle[name]
        if shuffle_permutation is None:
            token_indices[name] = pre_shuffle_idx
        else:
            try:
                token_indices[name] = shuffle_permutation.index(pre_shuffle_idx)
            except ValueError as exc:
                raise RuntimeError(
                    f"Raw feature {name!r} column {pre_shuffle_idx} is missing after feature shuffle."
                ) from exc

    return token_indices


def to_sample_token_repr(x: torch.Tensor) -> np.ndarray:
    # Keep one representation per sample.
    # Typical shapes observed at hooks:
    # - [batch, token, emb]
    # - [1, batch, token, emb] (extra leading axis from internal wrapper)
    x_np = x.detach().cpu().numpy()
    if x_np.ndim == 4:
        if x_np.shape[0] == 1:
            x_np = x_np[0]
        elif x_np.shape[1] == 1:
            x_np = x_np[:, 0]
        else:
            # Fallback: average leading axis (rare for n_estimators=1).
            x_np = x_np.mean(axis=0)
    elif x_np.ndim != 3:
        raise RuntimeError(f"Unexpected tensor rank for PCA representation: {x_np.shape}")
    return x_np


def extract_attention(
    model: TabPFNClassifier,
    x_test: np.ndarray,
    context: str,
    layers_1_based: list[int],
) -> tuple[list[dict], dict[str, np.ndarray], list[str]]:
    feature_rows: list[dict] = []
    stage_to_repr: dict[str, np.ndarray] = {}

    capture_layers = sorted(set([1] + layers_1_based))
    layer_buffers: dict[int, list[np.ndarray]] = {l: [] for l in layers_1_based}
    layer_inputs_by_sample: dict[int, dict[int, list[np.ndarray]]] = {l: {} for l in capture_layers}
    hooks = []
    token_labels: list[str] | None = None
    active_sample_ids: np.ndarray | None = None
    for l1 in capture_layers:
        module = model.models_[0].transformer_encoder.layers[l1 - 1].self_attn_between_features

        def make_hook(layer_id: int):
            def hook_fn(mod, inputs, _output):
                x = inputs[0].detach()
                x_np = to_sample_token_repr(x)
                nonlocal token_labels
                if token_labels is None:
                    token_labels = [f"token_{i+1}" for i in range(x_np.shape[1])]
                if active_sample_ids is None:
                    return
                n_take = min(len(active_sample_ids), x_np.shape[0])
                for k in range(n_take):
                    sid = int(active_sample_ids[k])
                    layer_inputs_by_sample[layer_id].setdefault(sid, []).append(x_np[k])

                if layer_id not in layer_buffers:
                    return
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
        j = min(i + PRED_BATCH_SIZE, len(x_test))
        active_sample_ids = np.arange(i, j, dtype=int)
        _ = model.predict(x_test[i : i + PRED_BATCH_SIZE])
    active_sample_ids = None
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

    for l1 in capture_layers:
        by_sid = layer_inputs_by_sample[l1]
        if len(by_sid) == 0:
            continue
        stage_name = "Input" if l1 == 1 else f"L{l1}"
        # Keep exactly one representation per requested test sample.
        # If a sample was captured multiple times internally, average duplicates.
        if token_labels is None:
            raise RuntimeError("Missing token labels while building stage representations.")
        n_tok = len(token_labels)
        emb_dim = next(iter(by_sid.values()))[0].shape[-1]
        arr = np.full((len(x_test), n_tok, emb_dim), np.nan, dtype=np.float32)
        for sid, reps in by_sid.items():
            if sid < 0 or sid >= len(x_test):
                continue
            arr[sid] = np.mean(np.stack(reps, axis=0), axis=0)
        stage_to_repr[stage_name] = arr

    if token_labels is None:
        token_labels = ["token_1"]
    return feature_rows, stage_to_repr, token_labels


def build_pca_rows(
    repr_full: dict[str, np.ndarray],
    repr_b10: dict[str, np.ndarray],
    raw_tokens_full: dict[str, int],
    raw_tokens_b10: dict[str, int],
    y_bin_by_sample: np.ndarray,
) -> pd.DataFrame:
    stage_order = ["Input"] + [f"L{x}" for x in LAYERS]
    rows: list[dict] = []
    for stage in stage_order:
        if stage not in repr_full or stage not in repr_b10:
            continue
        arr_full = repr_full[stage]
        arr_b10 = repr_b10[stage]
        n_full, n_tok, emb_dim = arr_full.shape
        n_b10 = arr_b10.shape[0]
        n_common = min(n_full, n_b10, len(y_bin_by_sample))
        if n_common < 2:
            continue
        arr_full = arr_full[:n_common]
        arr_b10 = arr_b10[:n_common]
        y_bin_stage = y_bin_by_sample[:n_common]

        feat_labels = FEATURES
        full_idx = [raw_tokens_full[name] for name in feat_labels]
        b10_idx = [raw_tokens_b10[name] for name in feat_labels]
        if max(full_idx) >= n_tok or max(b10_idx) >= arr_b10.shape[1]:
            raise RuntimeError(
                f"Raw token indices exceed captured token count at stage={stage}: "
                f"full={full_idx}, b10={b10_idx}, n_tok={n_tok}, b10_n_tok={arr_b10.shape[1]}"
            )
        # One point per sample per true raw input attribute token.
        feat_full = arr_full[:, full_idx, :]
        feat_b10 = arr_b10[:, b10_idx, :]
        n_feat = feat_full.shape[1]

        emb_full = feat_full.reshape(n_common * n_feat, emb_dim)
        emb_b10 = feat_b10.reshape(n_common * n_feat, emb_dim)

        x_all = np.concatenate([emb_full, emb_b10], axis=0)
        valid = np.isfinite(x_all).all(axis=1)
        if valid.sum() < 2:
            continue
        x_use = x_all[valid].astype(np.float64, copy=False)
        x_use = sk.preprocessing.StandardScaler().fit_transform(x_use)
        pca = sk.decomposition.PCA(n_components=2, random_state=ENS_SEED)
        pcs = pca.fit_transform(x_use)

        n_flat = n_common * n_feat
        pc_full = np.full((n_flat, 2), np.nan, dtype=float)
        pc_b10 = np.full((n_flat, 2), np.nan, dtype=float)
        pc_full_valid = valid[:n_flat]
        pc_b10_valid = valid[n_flat:]
        pc_full[pc_full_valid, :] = pcs[: pc_full_valid.sum(), :]
        pc_b10[pc_b10_valid, :] = pcs[pc_full_valid.sum():, :]
        pc_full = pc_full.reshape(n_common, n_feat, 2)
        pc_b10 = pc_b10.reshape(n_common, n_feat, 2)

        for i in range(n_common):
            yb = int(y_bin_stage[i])
            for f in range(n_feat):
                rows.append(
                    {
                        "context": "full",
                        "stage": stage,
                        "sample_id": int(i),
                        "token": feat_labels[f] if f < len(feat_labels) else f"feature_{f + 1}",
                        "y_bin": f"C{yb + 1}",
                        "pc1": float(pc_full[i, f, 0]),
                        "pc2": float(pc_full[i, f, 1]),
                        "explained_var_pc1": float(pca.explained_variance_ratio_[0]),
                        "explained_var_pc2": float(pca.explained_variance_ratio_[1]),
                    }
                )
        for i in range(n_common):
            yb = int(y_bin_stage[i])
            for f in range(n_feat):
                rows.append(
                    {
                        "context": "b10",
                        "stage": stage,
                        "sample_id": int(i),
                        "token": feat_labels[f] if f < len(feat_labels) else f"feature_{f + 1}",
                        "y_bin": f"C{yb + 1}",
                        "pc1": float(pc_b10[i, f, 0]),
                        "pc2": float(pc_b10[i, f, 1]),
                        "explained_var_pc1": float(pca.explained_variance_ratio_[0]),
                        "explained_var_pc2": float(pca.explained_variance_ratio_[1]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
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

    y_train_bin, bin_edges = quantile_bin_labels(y_train, N_TARGET_BINS)
    y_test_bin = apply_bin_edges(df_test[TARGET], bin_edges)
    y_test_bin_attn = y_test_bin[idx_test_attn]
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
    raw_tokens_full = raw_attribute_token_indices(model_full, FEATURES)
    raw_tokens_b10 = raw_attribute_token_indices(model_b10, FEATURES)
    pd.DataFrame(
        [
            {
                "context": "full",
                "feature": name,
                "token": f"token_{raw_tokens_full[name] + 1}",
                "token_index_0based": raw_tokens_full[name],
            }
            for name in FEATURES
        ]
        + [
            {
                "context": "b10",
                "feature": name,
                "token": f"token_{raw_tokens_b10[name] + 1}",
                "token_index_0based": raw_tokens_b10[name],
            }
            for name in FEATURES
        ]
    ).to_csv(OUT_RAW_TOKEN_MAP, index=False)

    feature_rows_full, repr_full, _ = extract_attention(model_full, X_test_attn, "full", layers_used)
    feature_rows_b10, repr_b10, _ = extract_attention(model_b10, X_test_attn, "b10", layers_used)
    feature_rows = feature_rows_full + feature_rows_b10
    pd.DataFrame(feature_rows).to_csv(OUT_FEATURE_LONG, index=False)
    pca_rows = build_pca_rows(
        repr_full=repr_full,
        repr_b10=repr_b10,
        raw_tokens_full=raw_tokens_full,
        raw_tokens_b10=raw_tokens_b10,
        y_bin_by_sample=y_test_bin_attn,
    )
    pca_rows.to_csv(OUT_FEATURE_TOKEN_PCA_LONG, index=False)

    print("Classifier checkpoint attention extraction complete.")
    print(f"Wrote: {OUT_FEATURE_LONG}")
    print(f"Wrote: {OUT_FEATURE_TOKEN_PCA_LONG}")
    print(f"Wrote: {OUT_RAW_TOKEN_MAP}")
    print("For Fig. 4 (incl. panel (c) PCA), run Code/4.4.Fig.4.R after this script.")


if __name__ == "__main__":
    main()
