#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
3.4.mininalExample.py — Attribute-token PCA (Figure 3 style).

**What this script does**
  Build **N × d** embedding rows (**d = len(FEATURES)** feature tokens only; label column excluded),
  run **per-panel** ``StandardScaler`` + ``PCA(2)``, scatter with **one color per feature**.
  **Input** panel = tensor into layer 1 ``self_attn_between_features`` (**forward hook on
  ``inputs[0]``**). **Depth panels** = encoder **block outputs** after **{1, 2, 3, 6, 9, 12}**,
  sliced as **train rows × first d token columns** (TabPFN ``(R,C,E)`` grid). All hooks fire in a
  **single** ``get_embeddings`` forward (**no** separate ``predict`` for Input). Writes
  ``tabpfn_attribute_tokens_evolution.png`` and a small diagnostic PNG.

**Compared to the git ``main`` minimal version** (``tabpfn_embedding_train_side_by_side`` style)
  - **Quantity plotted:** git version PCA’d **one label-token embedding per train row** (library
    slice ``[:,:,-1]``), colored by **target quantile bin**. Here we PCA **all attribute-token
    vectors** (**N×d** rows), colored by **which feature**.
  - **Depth set:** git used hooks on **{3, 6, 9, 12}** only. Here **{1, 2, 3, 6, 9, 12}** plus a
    separate **Input** tensor (pre–between-features-attention at L1).
  - **Input definition:** git had no Input hook. Here Input matches **3.3-style** hook site:
    ``transformer_encoder.layers[0].self_attn_between_features``, ``inputs[0]``.
  - **Tensor geometry:** extraction uses **row/column** slicing on block tensors (Eq.~(1)-style),
    not a flat ``d×N`` sequence along one axis.
  - **Preprocessor-aligned columns for Input:** ``raw_attribute_token_indices`` maps **FEATURES**
    names to token indices after TabPFN preprocessing (shuffle / reshape steps).
  - **Memory / device:** default ``TABPFN_DEVICE`` is **cpu** (env override e.g.
    ``TABPFN_DEVICE=auto``) after **MPS OOM** on full train forwards; hook only keeps
    ``inputs[0].detach().cpu()`` with **overwrite**, NumPy once after forward.
  - **Batch quirk:** if Input hook sees batch **2N**, we **take first N rows** (TabPFN behavior on
    some forwards) and print a note.
  - **Outputs / layout:** main figure is a **2×4** grid (Input + six block depths + legend); git was
    **4 depth panels + bottom official vs L12** on a different filename.

Requires ``transformer_encoder.layers`` on the checkpoint for the Input hook.
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
OUT_FIG = os.path.join(OUT_DIR, "tabpfn_attribute_tokens_evolution.png")

TRAIN_YEAR = 2024
TARGET = "yH"
# TabPFN uses one token per column; extend this list to explore deeper embeddings (keep names as in INPUT_FILE).
# xP/xS/y targets are Ghc-normalized in ``main``; SZA is cos(rad).
FEATURES = [
    "xP",
    "xS",
    "SZA",
    "lcc",
    "mcc",
    "tcsw",
    "tcwv",
    "t2m",
    "sp",
    "hcc",
    "W",
    "rh",
]
N_TARGET_BINS = 7
N_TRAIN = 2000
RANDOM_STATE = 123

# Full train forward + hooks exceeds Apple MPS budget for large N; use CPU by default.
# Override: ``TABPFN_DEVICE=auto`` or ``TABPFN_DEVICE=mps`` / ``cuda`` when it fits.
TABPFN_DEVICE = os.environ.get("TABPFN_DEVICE", "cpu")

EXPECTED_N_ENCODER_LAYERS = 12
LAYERS_DEPTH_ROW = [1, 2, 3, 6, 9, 12]

D_ATTR = len(FEATURES)
ATTRIBUTE_NAMES = FEATURES


def _attribute_color_map(n_attr: int) -> np.ndarray:
    """Distinct colors for arbitrary ``d`` (Set1 only has a few categories)."""
    if n_attr <= 10:
        return plt.cm.tab10(np.linspace(0.05, 0.95, n_attr))
    if n_attr <= 20:
        return plt.cm.tab20(np.linspace(0.05, 0.95, n_attr))
    return plt.cm.nipy_spectral(np.linspace(0.1, 0.9, n_attr))


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
            f"Expected exactly {expect_n_layers} encoder layers; got {n}."
        )
    if layer_1_based > n:
        raise RuntimeError(f"Asked for L{layer_1_based} but stack has only {n} layers.")
    return stack[idx]


def attribute_train_matrix_n_times_d(
    x: torch.Tensor,
    *,
    n_train_labels: int,
    num_thinking_rows: int,
    n_attributes: int,
) -> np.ndarray:
    """Train rows × first ``n_attributes`` token cols → ``(N_train * n_attributes, E)``."""
    x = x.detach().float().cpu()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise RuntimeError(f"Expected (B,R,C,E) with B=1 or (R,C,E), got {tuple(x.shape)}")
    rs = num_thinking_rows
    re = rs + n_train_labels
    if x.shape[0] < re or x.shape[1] < n_attributes + 1:
        raise RuntimeError(
            f"Tensor (R,C,E)={tuple(x.shape)} incompatible with "
            f"n_train={n_train_labels}, n_think={num_thinking_rows}, d={n_attributes} (+ label)."
        )
    attrs = x[rs:re, :n_attributes, :]
    out = attrs.reshape(n_train_labels * n_attributes, -1).numpy()
    return np.nan_to_num(out.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def num_thinking_rows_from_arch(arch: torch.nn.Module) -> int:
    if hasattr(arch, "add_thinking_rows"):
        return int(arch.add_thinking_rows.num_thinking_rows)
    tok = getattr(arch, "add_thinking_tokens", None)
    if tok is not None:
        return int(tok.num_thinking_rows)
    return 0


def n_train_rows_from_clf(clf: TabPFNClassifier, X_fit: np.ndarray) -> int:
    ex = clf.executor_
    if hasattr(ex, "X_train"):
        return int(ex.X_train.shape[0])
    if hasattr(ex, "X_train_shape_before_preprocessing"):
        return int(ex.X_train_shape_before_preprocessing[0])
    return int(X_fit.shape[0])


def raw_attribute_token_indices(model: TabPFNClassifier, feature_names: list[str]) -> dict[str, int]:
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


def hook_input_cpu_to_bte(last_cpu: torch.Tensor) -> np.ndarray:
    """``inputs[0].detach().cpu()`` → ``(batch, token, emb)`` float32."""
    t = last_cpu.numpy()
    if t.ndim == 4:
        if t.shape[0] == 1:
            t = t[0]
        elif t.shape[1] == 1:
            t = t[:, 0]
        else:
            t = t.mean(axis=0)
    if t.ndim != 3:
        raise RuntimeError(f"Expected (batch, token, emb); got {t.shape}")
    return np.asarray(t, dtype=np.float32)


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
        device=TABPFN_DEVICE,
    )
    clf.fit(X_train_s, y_train_bin)

    TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)

    arch = clf.models_[0]
    te = getattr(arch, "transformer_encoder", None)
    if te is None:
        raise RuntimeError(
            "Need ``transformer_encoder.layers[0].self_attn_between_features`` for Input capture."
        )
    inp_mod = te.layers[0].self_attn_between_features

    captured_by_layer: dict[int, torch.Tensor] = {}
    last_inp_cpu: torch.Tensor | None = None

    def _input_hook(_m: torch.nn.Module, inp: tuple, _out: torch.Tensor) -> None:
        nonlocal last_inp_cpu
        # Overwrite only — submodule may run multiple times per forward.
        last_inp_cpu = inp[0].detach().cpu()

    def _make_capture(layer_id: int):
        def _fn(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            captured_by_layer[layer_id] = out.detach()

        return _fn

    hooks: list = [inp_mod.register_forward_hook(_input_hook)]
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
    n_think = num_thinking_rows_from_arch(arch)
    n_x = len(X_train_s)

    tok_ix = raw_attribute_token_indices(clf, FEATURES)
    attr_cols = [tok_ix[name] for name in FEATURES]
    need_tok = max(attr_cols) + 1

    if last_inp_cpu is None:
        raise RuntimeError("Input hook did not fire during get_embeddings.")
    t_in = hook_input_cpu_to_bte(last_inp_cpu)
    if t_in.shape[1] < need_tok:
        raise RuntimeError(f"Input tensor has {t_in.shape[1]} tokens; need >= {need_tok}.")
    b_in = int(t_in.shape[0])
    if b_in == n_x:
        pass
    elif b_in == 2 * n_x:
        # Observed on some TabPFN get_embeddings forwards: hook sees duplicated batch (e.g. 2N).
        print(f"Note: Input hook batch was {b_in} (2×N); using first N={n_x} rows for attribute PCA.")
        t_in = t_in[:n_x]
    else:
        raise RuntimeError(
            f"Input hook batch dim {b_in} is neither N={n_x} nor 2N={2 * n_x}. "
            "Cannot align one embedding row per training instance."
        )
    sel_in = t_in[:, attr_cols, :]
    layer_tokens: dict[str | int, np.ndarray] = {
        "input": np.nan_to_num(
            sel_in.reshape(n_x * D_ATTR, -1).astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
    }

    for lid in LAYERS_DEPTH_ROW:
        captured_by_layer[lid] = captured_by_layer[lid].detach().cpu()
        layer_tokens[lid] = attribute_train_matrix_n_times_d(
            captured_by_layer[lid],
            n_train_labels=n_train_fit,
            num_thinking_rows=n_think,
            n_attributes=D_ATTR,
        )

    print(f"Training instances N={n_train_fit}, d={D_ATTR}, thinking rows={n_think}")
    print(f"Input token matrix shape: {layer_tokens['input'].shape}")

    attribute_colors = np.tile(np.arange(D_ATTR, dtype=int), n_train_fit)
    colors_map = _attribute_color_map(D_ATTR)

    fig = plt.figure(figsize=(18, 9), layout="constrained")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1])

    plot_positions_top: list[str | int] = ["input"] + LAYERS_DEPTH_ROW[:3]
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for ax, layer_name in zip(axes_top, plot_positions_top, strict=True):
        if layer_name not in layer_tokens:
            continue
        tokens = layer_tokens[layer_name]
        scaler_pca = StandardScaler()
        tokens_std = scaler_pca.fit_transform(tokens)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        tokens_pca = pca.fit_transform(tokens_std)

        for attr_idx in range(D_ATTR):
            mask = attribute_colors == attr_idx
            ax.scatter(
                tokens_pca[mask, 0],
                tokens_pca[mask, 1],
                c=[colors_map[attr_idx]],
                label=ATTRIBUTE_NAMES[attr_idx],
                alpha=0.6,
                s=8,
                rasterized=True,
            )

        ev = pca.explained_variance_ratio_
        if layer_name == "input":
            title = "Input (L1 self_attn_between_features in)"
        else:
            title = f"L{layer_name} (block out)"
        ax.set_title(f"{title}\n(PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(4)]
    mid_layers = LAYERS_DEPTH_ROW[3:6]

    for i, ax in enumerate(axes_bottom):
        if i < len(mid_layers) and mid_layers[i] in layer_tokens:
            lid = mid_layers[i]
            tokens = layer_tokens[lid]
            scaler_pca = StandardScaler()
            tokens_std = scaler_pca.fit_transform(tokens)
            pca_bl = PCA(n_components=2, random_state=RANDOM_STATE)
            tokens_pca = pca_bl.fit_transform(tokens_std)
            for attr_idx in range(D_ATTR):
                mask = attribute_colors == attr_idx
                ax.scatter(
                    tokens_pca[mask, 0],
                    tokens_pca[mask, 1],
                    c=[colors_map[attr_idx]],
                    alpha=0.6,
                    s=8,
                    rasterized=True,
                )
            ev = pca_bl.explained_variance_ratio_
            ax.set_title(f"L{lid} (block out)\n(PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        elif i == 3:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors_map[j],
                    markersize=8,
                    label=ATTRIBUTE_NAMES[j],
                )
                for j in range(D_ATTR)
            ]
            ax.legend(
                handles=handles,
                loc="center",
                frameon=True,
                title="Attributes",
                ncol=2 if D_ATTR > 8 else 1,
                fontsize=max(5, 9 - D_ATTR // 4),
            )
            ax.axis("off")
        else:
            ax.axis("off")

    fig.suptitle(
        "Attribute token embeddings (N×d, first d feature tokens)\n"
        "Input = L1 self_attn_between_features inputs[0]; other panels = encoder block outputs",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUT_FIG}")

    # Diagnostic: official train embedding vs last-layer attribute PCA (optional)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    emb_arr = emb[0] if isinstance(emb, np.ndarray) and emb.ndim == 3 else emb
    if isinstance(emb_arr, np.ndarray) and emb_arr.ndim == 2 and emb_arr.shape[0] == n_x:
        Z = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
            StandardScaler().fit_transform(np.nan_to_num(emb_arr.astype(np.float64)))
        )
        ax1.scatter(Z[:, 0], Z[:, 1], c=y_train_bin, cmap="viridis", alpha=0.6, s=8)
        ax1.set_title(f"Official get_embeddings (train)\n{emb_arr.shape}")
    else:
        ax1.text(0.5, 0.5, "emb shape not (N, E)", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Official get_embeddings")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    last_layer = LAYERS_DEPTH_ROW[-1]
    if last_layer in layer_tokens:
        tokens = layer_tokens[last_layer]
        tokens_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
            StandardScaler().fit_transform(tokens)
        )
        for attr_idx in range(D_ATTR):
            mask = attribute_colors == attr_idx
            ax2.scatter(
                tokens_pca[mask, 0],
                tokens_pca[mask, 1],
                c=[colors_map[attr_idx]],
                alpha=0.5,
                s=5,
            )
        ax2.set_title(f"L{last_layer} attribute tokens\n{tokens.shape}")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    diagnostic_fig = os.path.join(OUT_DIR, "tabpfn_official_vs_hook_attribute_tokens.png")
    fig2.savefig(diagnostic_fig, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved diagnostic comparison to {diagnostic_fig}")


if __name__ == "__main__":
    main()
