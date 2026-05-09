#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.4.mininalExample.py — minimal TabPFN official **train** embeddings + PCA plot.

Reads arranged data, samples N_TRAIN rows from the train year, fits TabPFNClassifier,
extracts **training** embeddings via tabpfn_extensions.TabPFNEmbedding API
(``model.get_embeddings(..., data_source="train")``), PCA scatter colored by target bins.

No test data. Writes one PNG under Data/Output/Diag/. Requires CHECKPOINT_FILE on disk.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

PROJECT_PATH = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
INPUT_FILE = os.path.join(PROJECT_PATH, "Data", "arranged15min.txt")
CHECKPOINT_FILE = os.path.join(PROJECT_PATH, "tabpfn-v2-classifier-gn2p4bpt.ckpt")
OUT_DIR = os.path.join(PROJECT_PATH, "Data", "Output", "Diag")
OUT_FIG = os.path.join(OUT_DIR, "tabpfn_embedding_train_pca.png")

TRAIN_YEAR = 2024
TARGET = "yH"
FEATURES = ["xP", "SZA", "lcc", "mcc", "tcsw", "tcwv"]
N_TARGET_BINS = 7
N_TRAIN = 2000
RANDOM_STATE = 123


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
    emb = clf.get_embeddings(X_train_s, data_source="train")
    if emb.ndim == 3:
        emb = emb[0]
    if emb.shape[0] != N_TRAIN:
        raise RuntimeError(f"Expected {N_TRAIN} train embeddings, got {emb.shape[0]}")

    Z = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        StandardScaler().fit_transform(
            np.nan_to_num(emb.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        )
    )

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y_train_bin, cmap="viridis", alpha=0.75, s=14, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Target quantile bin (train)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Official TabPFN train embeddings (PCA)")
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
