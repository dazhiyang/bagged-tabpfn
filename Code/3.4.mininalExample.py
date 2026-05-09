#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.4.mininalExample.py — minimal TabPFN official **train** embeddings + PCA plot.

Reads arranged data, samples N_TRAIN rows from the train year, fits TabPFNClassifier,
extracts training embeddings via ``TabPFNClassifier.get_embeddings(..., data_source="train")``
and hook reconstructions with **library row/column slices** after encoder layers **3, 6, 9, 12**.
**Top row:** four PCA panels (one scaler + PCA fit per layer). **Bottom row:** official vs L12
hook with shared scaler (official) + PCA (official), same as before.

Hook vs official:
  TabPFN **v2 clf with 12 layers**: official train/test embeddings equal slicing the
  tensor **after encoder layer L12** (1-based), i.e. ``blocks[11]`` or
  ``transformer_encoder.layers[11]`` **output** — last column ``-1``, same row ranges as
  the library. This script hooks **L12** explicitly and checks ``nlayers == 12``.
  A hook on ``self_attn_between_features`` **input** inside a layer (3.3) is different.

Writes one PNG (4-layer depth row + official vs L12 row) under Data/Output/Diag/.
Requires CHECKPOINT_FILE on disk.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

PROJECT_PATH = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
INPUT_FILE = os.path.join(PROJECT_PATH, "Data", "arranged15min.txt")
CHECKPOINT_FILE = os.path.join(PROJECT_PATH, "tabpfn-v2-classifier-gn2p4bpt.ckpt")
OUT_DIR = os.path.join(PROJECT_PATH, "Data", "Output", "Diag")
OUT_FIG = os.path.join(OUT_DIR, "tabpfn_embedding_train_side_by_side.png")

TRAIN_YEAR = 2024
TARGET = "yH"
FEATURES = ["xP", "SZA", "lcc", "mcc", "tcsw", "tcwv"]
N_TARGET_BINS = 7
N_TRAIN = 2000
RANDOM_STATE = 123

# TabPFN-v2 classifier checkpoint with 12 encoder layers.
EXPECTED_N_ENCODER_LAYERS = 12
LAYERS_DEPTH_ROW = [3, 6, 9, 12]
HOOK_LAYER_1_BASED = 12  # must be last layer; used for official vs hook sanity check


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


def encoder_block_at_layer_1_based(
    model: torch.nn.Module,
    layer_1_based: int,
    *,
    expect_n_layers: int,
) -> torch.nn.Module:
    """Return encoder block ``layer_1_based`` (1-based index). Must match ``expect_n_layers``."""
    idx = layer_1_based - 1
    if idx < 0:
        raise ValueError(f"layer_1_based must be >= 1, got {layer_1_based}")
    if hasattr(model, "blocks"):
        stack = model.blocks
    elif hasattr(model, "transformer_encoder"):
        stack = model.transformer_encoder.layers
    else:
        raise RuntimeError(
            "Cannot locate encoder stack (no ``blocks`` or ``transformer_encoder.layers``)."
        )
    n = len(stack)
    if n != expect_n_layers:
        raise RuntimeError(
            f"Expected exactly {expect_n_layers} encoder layers (TabPFN v2 12L checkpoint); "
            f"got {n}. Adjust EXPECTED_N_ENCODER_LAYERS or checkpoint."
        )
    if layer_1_based > n:
        raise RuntimeError(f"Asked for L{layer_1_based} but stack has only {n} layers.")
    return stack[idx]


def official_train_test_from_x_brcd(
    x_brcd: torch.Tensor,
    *,
    n_train_labels: int,
    num_thinking_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same row/column slicing as ``TabPFNV2p6.forward`` / ``TabPFNV2p5.forward``."""
    num_train_and_thinking = n_train_labels + num_thinking_rows
    train_rows_start = num_thinking_rows
    train_rows_end = num_train_and_thinking
    test_emb_mbd = x_brcd[:, num_train_and_thinking:, -1].transpose(0, 1)
    train_emb_nbd = x_brcd[:, train_rows_start:train_rows_end, -1].transpose(0, 1)
    return train_emb_nbd, test_emb_mbd


def num_thinking_rows_on_arch(arch: torch.nn.Module) -> int:
    if hasattr(arch, "add_thinking_rows"):
        return int(arch.add_thinking_rows.num_thinking_rows)
    tok = getattr(arch, "add_thinking_tokens", None)
    if tok is not None:
        return int(tok.num_thinking_rows)
    return 0


def n_train_rows_from_clf(clf: TabPFNClassifier, X_fit: np.ndarray) -> int:
    """Training row count for slicing (handles InferenceEngine vs cache-preprocessing executor)."""
    ex = clf.executor_
    if hasattr(ex, "X_train"):
        return int(ex.X_train.shape[0])
    if hasattr(ex, "X_train_shape_before_preprocessing"):
        return int(ex.X_train_shape_before_preprocessing[0])
    return int(X_fit.shape[0])


def main() -> None:
    if not os.path.isfile(CHECKPOINT_FILE):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}\n"
            "Download or point CHECKPOINT_FILE to your TabPFN clf checkpoint."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, sep="\t")
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df[["yH", "yL", "xS", "xP"]] = df[["yH", "yL", "xS", "xP"]].div(df["Ghc"], axis=0)
    df["SZA"] = np.cos(np.radians(df["SZA"]))

    df_train = df.loc[df["Time"].dt.year == TRAIN_YEAR].copy()
    rng = np.random.default_rng(RANDOM_STATE)
    if len(df_train) < N_TRAIN:
        raise RuntimeError(f"Need at least {N_TRAIN} train-year rows, got {len(df_train)}")

    tr_idx = rng.choice(len(df_train), size=N_TRAIN, replace=False)
    X_train = df_train.iloc[tr_idx][FEATURES].to_numpy(dtype=float)
    y_train_ser = df_train.iloc[tr_idx][TARGET]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    y_train_bin, _edges = quantile_bin_labels(y_train_ser, N_TARGET_BINS)

    clf = TabPFNClassifier(
        model_path=CHECKPOINT_FILE,
        n_estimators=1,
        ignore_pretraining_limits=True,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_s, y_train_bin)

    TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)

    arch = clf.models_[0]
    captured_by_layer: dict[int, torch.Tensor] = {}

    def _make_capture(layer_id: int):
        def _fn(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            captured_by_layer[layer_id] = out.detach()

        return _fn

    hooks: list = []
    for lid in LAYERS_DEPTH_ROW:
        mod = encoder_block_at_layer_1_based(
            arch,
            lid,
            expect_n_layers=EXPECTED_N_ENCODER_LAYERS,
        )
        hooks.append(mod.register_forward_hook(_make_capture(lid)))
    try:
        emb = clf.get_embeddings(X_train_s, data_source="train")
    finally:
        for h in hooks:
            h.remove()

    if set(captured_by_layer.keys()) != set(LAYERS_DEPTH_ROW):
        raise RuntimeError(
            f"Missing layer captures: expected {LAYERS_DEPTH_ROW}, got {sorted(captured_by_layer)}"
        )

    n_train_fit = n_train_rows_from_clf(clf, X_train_s)
    n_think = num_thinking_rows_on_arch(arch)

    train_by_layer: dict[int, np.ndarray] = {}
    for lid in LAYERS_DEPTH_ROW:
        train_bne, _te = official_train_test_from_x_brcd(
            captured_by_layer[lid],
            n_train_labels=n_train_fit,
            num_thinking_rows=n_think,
        )
        train_by_layer[lid] = train_bne.squeeze(1).cpu().numpy()

    train_hook = train_by_layer[HOOK_LAYER_1_BASED]

    if emb.ndim == 3:
        emb = emb[0]
    if emb.shape[0] != N_TRAIN:
        raise RuntimeError(f"Expected {N_TRAIN} train embeddings, got {emb.shape[0]}")

    max_abs = float(np.max(np.abs(emb.astype(np.float64) - train_hook.astype(np.float64))))
    print(
        f"Official train emb vs L{HOOK_LAYER_1_BASED} block output + library slices, "
        f"max |diff|: {max_abs:.6e}"
    )

    emb_off = np.nan_to_num(emb.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    emb_hook = np.nan_to_num(train_hook.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    emb_scaler = StandardScaler()
    emb_off_s = emb_scaler.fit_transform(emb_off)
    emb_hook_s = emb_scaler.transform(emb_hook)

    pca_bottom = PCA(n_components=2, random_state=RANDOM_STATE)
    Z_off = pca_bottom.fit_transform(emb_off_s)
    Z_hook = pca_bottom.transform(emb_hook_s)

    fig = plt.figure(figsize=(14.5, 8.5), layout="constrained")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.05])
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_official = fig.add_subplot(gs[1, 0:2])
    ax_l12_compare = fig.add_subplot(gs[1, 2:4])

    sc_depth = None
    for ax, lid in zip(axes_top, LAYERS_DEPTH_ROW, strict=True):
        raw = np.nan_to_num(
            train_by_layer[lid].astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        zs = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
            StandardScaler().fit_transform(raw)
        )
        sc_depth = ax.scatter(
            zs[:, 0],
            zs[:, 1],
            c=y_train_bin,
            cmap="viridis",
            alpha=0.75,
            s=12,
            rasterized=True,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"L{lid} output + library slices")

    sc_bottom = None
    titles_bottom = (
        "Official get_embeddings (train)",
        f"L{HOOK_LAYER_1_BASED} block output + library slices",
    )
    for ax, Z, title in zip(
        (ax_official, ax_l12_compare),
        (Z_off, Z_hook),
        titles_bottom,
        strict=True,
    ):
        sc_bottom = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=y_train_bin,
            cmap="viridis",
            alpha=0.75,
            s=14,
            rasterized=True,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)

    assert sc_depth is not None and sc_bottom is not None
    fig.colorbar(sc_depth, ax=[*axes_top], location="right", shrink=0.82, label="Target quantile bin (train)")
    fig.colorbar(sc_bottom, ax=[ax_official, ax_l12_compare], location="right", shrink=0.88, label="Target quantile bin (train)")
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
