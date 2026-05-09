#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
3.3.bagged.py

Classifier-checkpoint stage embeddings + PCA rows for Fig. 3 panel (d):
- Load TabPFN v2 classifier from ``CHECKPOINT_FILE`` (local path; must exist).
- Bin normalized ``TARGET`` with **fixed intervals** on ``[0, 1.1]`` (see ``target_fixed_bins``);
  rows outside that union are **dropped** before fitting. Bootstrap indices still follow ``ENS_K``
  × ``COMBO_ORDER`` (member B10, combo ``COMBO``).
- **PCA embeddings**: same rows for Full vs B10 (``build_pca_rows``); **per stage**, fit
  **separate** ``StandardScaler`` + ``PCA(2)`` on Full rows and on B10 rows (axes differ by context).
- Writes ``feature_token_pca_layers_long.csv`` for ``Code/4.3.Fig.3.R``.

Attention matrices (panels (b)–(c)): run ``Code/3.5.attention.py`` first (chunked
``predict``); it writes ``attention_feature_layers_long.csv``.

Device: ``TABPFN_EMBED_DEVICE`` (default ``cpu``) for fit and ``get_embeddings``.
"""

import os
import numpy as np
import pandas as pd
import sklearn as sk
import torch
from tabpfn import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

PROJECT_PATH = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
INPUT_FILE = os.path.join(PROJECT_PATH, "Data", "arranged15min.txt")
DIAG_DIR = os.path.join(PROJECT_PATH, "Data", "Output", "Diag")
CHECKPOINT_FILE = os.path.join(PROJECT_PATH, "tabpfn-v2-classifier-gn2p4bpt.ckpt")

TRAIN_YEAR = 2024
TARGET = "yH"
FEATURES = ["xP", "SZA", "lcc", "mcc", "tcsw", "tcwv"]
COMBO = "yHxP"

N_TARGET_CLASSES = 3  # fixed bins on normalized yH; must match ``target_fixed_bins``
N_ESTIMATORS = 1
LAYERS = [1, 2, 3, 6, 9, 12]

ENS_K = 10
ENS_SEED = 123
BOOTSTRAP_N = 2000
B10_MEMBER_INDEX_1_BASED = 10
COMBO_ORDER = ["yHxP", "yHxS", "yLxP", "yLxS"]

TABPFN_EMBED_DEVICE = os.environ.get("TABPFN_EMBED_DEVICE", "cpu")

OUT_FEATURE_TOKEN_PCA_LONG = os.path.join(DIAG_DIR, "feature_token_pca_layers_long.csv")


def target_fixed_bins(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Classes 0,1,2 for ``[0, 0.3)``, ``[0.3, 0.9)``, ``[0.9, 1.1]`` on normalized ``TARGET``; ``keep`` masks rows to retain."""
    yv = y.to_numpy(dtype=float)
    m0 = (yv >= 0.0) & (yv < 0.3)
    m1 = (yv >= 0.3) & (yv < 0.9)
    m2 = (yv >= 0.9) & (yv <= 1.1)
    keep = m0 | m1 | m2
    lab = np.full(len(yv), -1, dtype=np.int64)
    lab[m0] = 0
    lab[m1] = 1
    lab[m2] = 2
    return lab, keep


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


def encoder_block_at_layer_1_based(model_arch: torch.nn.Module, layer_1_based: int) -> torch.nn.Module:
    idx = layer_1_based - 1
    if idx < 0:
        raise ValueError(f"layer_1_based must be >= 1, got {layer_1_based}")
    if hasattr(model_arch, "transformer_encoder"):
        stack = model_arch.transformer_encoder.layers
    elif hasattr(model_arch, "blocks"):
        stack = model_arch.blocks
    else:
        raise RuntimeError(
            "Cannot locate encoder stack (no ``transformer_encoder.layers`` or ``blocks``)."
        )
    if idx >= len(stack):
        raise RuntimeError(f"Asked for L{layer_1_based} but stack has only {len(stack)} layers.")
    return stack[idx]


def hook_input_cpu_to_bte(last_cpu: torch.Tensor) -> np.ndarray:
    """``inputs[0].detach().cpu()`` → ``(*, token, emb)`` float32."""
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


def extract_stage_embeddings_v34(
    model: TabPFNClassifier,
    X_query: np.ndarray,
    layers_1_based: list[int],
    feature_names: list[str],
    *,
    train_row_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Stage tensors for PCA CSV via ``get_embeddings(..., data_source=\"train\")`` + hooks.

    Slices the **train** row block from each tensor, then ``train_row_indices`` (length
    ``len(X_query)``): global indices for the full model, positions ``0..N_b10-1`` for B10.

    **Alignment with ``3.4.mininalExample.py`` (extraction only):**
    - **Input:** rows ``train_row_indices`` × token columns
      ``raw_attribute_token_indices`` in ``feature_names`` order → shape ``(M, d, emb)``.
    - **Depth (block outputs):** same rows × **first** ``d`` token columns (positional),
      ``d = len(feature_names)`` — matches ``attribute_train_matrix_n_times_d`` / ``[:d]``.
    """
    TabPFNEmbedding(tabpfn_clf=model, n_fold=0)
    arch = model.models_[0]
    te = getattr(arch, "transformer_encoder", None)
    if te is None:
        raise RuntimeError(
            "Need ``transformer_encoder.layers`` for Input + block embedding hooks."
        )

    n_query = int(X_query.shape[0])
    n_train_fit = n_train_rows_from_clf(model, X_query)
    n_think = num_thinking_rows_from_arch(arch)
    tok_ix = raw_attribute_token_indices(model, feature_names)
    attr_cols = [tok_ix[name] for name in feature_names]
    need_tok_in = max(attr_cols) + 1
    d_attr = len(feature_names)

    inp_mod = te.layers[0].self_attn_between_features
    captured_by_layer: dict[int, torch.Tensor] = {}
    last_inp_cpu: torch.Tensor | None = None

    def _input_hook(_m: torch.nn.Module, inp: tuple, _out: torch.Tensor) -> None:
        nonlocal last_inp_cpu
        last_inp_cpu = inp[0].detach().cpu()

    def _make_capture(layer_id: int):
        def _fn(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            captured_by_layer[layer_id] = out.detach()

        return _fn

    hooks: list = [inp_mod.register_forward_hook(_input_hook)]
    for lid in layers_1_based:
        hooks.append(
            encoder_block_at_layer_1_based(arch, lid).register_forward_hook(_make_capture(lid))
        )

    try:
        _ = model.get_embeddings(X_query, data_source="train")
    finally:
        for h in hooks:
            h.remove()

    if last_inp_cpu is None:
        raise RuntimeError("Input hook did not fire during get_embeddings.")
    if set(captured_by_layer.keys()) != set(layers_1_based):
        raise RuntimeError(
            f"Missing block captures: expected {sorted(layers_1_based)}, "
            f"got {sorted(captured_by_layer.keys())}"
        )

    rs_tr = n_think
    re_tr = n_think + n_train_fit
    t_in = hook_input_cpu_to_bte(last_inp_cpu)

    idx = np.asarray(train_row_indices, dtype=np.int64)
    if idx.ndim != 1 or idx.shape[0] != n_query:
        raise RuntimeError(
            f"train_row_indices must be 1D of length len(X_query)={n_query}; got shape {idx.shape}."
        )
    if idx.min() < 0 or idx.max() >= n_train_fit:
        raise RuntimeError(
            f"train_row_indices out of range for this model's N_train_fit={n_train_fit}: "
            f"[{idx.min()}, {idx.max()}]"
        )

    def input_train_block(t: np.ndarray) -> np.ndarray:
        if t.ndim != 3:
            raise RuntimeError(f"Expected (*, token, emb); got {t.shape}")
        if t.shape[0] >= re_tr:
            blk = np.asarray(t[rs_tr:re_tr], dtype=np.float32)
        elif t.shape[0] == n_train_fit:
            blk = np.asarray(t, dtype=np.float32)
        elif t.shape[0] == 2 * n_train_fit:
            print(
                f"Note: embedding Input hook rows={t.shape[0]} (2×N_train); "
                f"using first N_train={n_train_fit}."
            )
            blk = np.asarray(t[:n_train_fit], dtype=np.float32)
        else:
            raise RuntimeError(
                f"Cannot align Input tensor rows {t.shape[0]} with "
                f"train block ending at {re_tr} or N_train={n_train_fit}."
            )
        if blk.shape[0] != n_train_fit:
            raise RuntimeError(
                f"Expected Input train block with N_train={n_train_fit}; got {blk.shape[0]} rows."
            )
        return blk

    def block_train_matrix(x: torch.Tensor) -> np.ndarray:
        xf = x.detach().float().cpu()
        if xf.ndim == 4 and xf.shape[0] == 1:
            xf = xf[0]
        if xf.ndim != 3:
            raise RuntimeError(f"Expected block output (R,C,E); got {tuple(x.shape)}")
        if xf.shape[0] < re_tr:
            raise RuntimeError(
                f"Block tensor rows {xf.shape[0]} < train slice end {re_tr} "
                f"(thinking={n_think}, N_train_fit={n_train_fit})."
            )
        blk = np.asarray(xf[rs_tr:re_tr], dtype=np.float32)
        if blk.shape[0] != n_train_fit:
            raise RuntimeError(
                f"Expected block train rows N_train={n_train_fit}; got {blk.shape[0]}."
            )
        return blk

    full_in = input_train_block(t_in)
    in_rows = full_in[idx]
    blocks_np: dict[int, np.ndarray] = {}
    for lid in layers_1_based:
        blk = block_train_matrix(captured_by_layer[lid])
        if blk.shape[1] < d_attr:
            raise RuntimeError(
                f"Block L{lid} has {blk.shape[1]} tokens; need >= d={d_attr} "
                "(first-d columns, 3.4-style)."
            )
        blocks_np[lid] = blk[idx][:, :d_attr, :]

    if in_rows.shape[1] < need_tok_in:
        raise RuntimeError(
            f"Input tensor has {in_rows.shape[1]} tokens; need >= {need_tok_in} for raw attributes."
        )

    in_md = in_rows[:, attr_cols, :]
    emb_dim = int(in_md.shape[2])

    def clean_md(sl: np.ndarray) -> np.ndarray:
        return np.nan_to_num(sl.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

    stage_to_repr: dict[str, np.ndarray] = {"Input": clean_md(in_md)}
    for lid in sorted(layers_1_based):
        stage_to_repr[f"L{lid}"] = clean_md(blocks_np[lid])

    if stage_to_repr["Input"].shape != (n_query, d_attr, emb_dim):
        raise RuntimeError(
            f"Expected Input (M,d,E)=({n_query},{d_attr},{emb_dim}); got {stage_to_repr['Input'].shape}."
        )
    for lid in sorted(layers_1_based):
        shp = stage_to_repr[f"L{lid}"].shape
        if shp != (n_query, d_attr, emb_dim):
            raise RuntimeError(
                f"Expected L{lid} (M,d,E)=({n_query},{d_attr},{emb_dim}); got {shp}."
            )

    return stage_to_repr


def _pca_flat_embeddings(
    emb: np.ndarray,
    n_common: int,
    n_feat: int,
) -> tuple[np.ndarray | None, sk.decomposition.PCA | None]:
    """``emb`` shape ``(n_common * n_feat, emb_dim)`` → grid ``(n_common, n_feat, 2)`` or None if <2 finite rows."""
    n_flat = n_common * n_feat
    if emb.shape[0] != n_flat:
        raise RuntimeError(f"Expected emb rows {n_flat}, got {emb.shape[0]}")
    valid = np.isfinite(emb).all(axis=1)
    if valid.sum() < 2:
        return None, None
    x = emb[valid].astype(np.float64, copy=False)
    x = sk.preprocessing.StandardScaler().fit_transform(x)
    pca = sk.decomposition.PCA(n_components=2, random_state=ENS_SEED)
    pcs = pca.fit_transform(x)
    flat = np.full((n_flat, 2), np.nan, dtype=float)
    flat[valid, :] = pcs
    return flat.reshape(n_common, n_feat, 2), pca


def build_pca_rows(
    repr_full: dict[str, np.ndarray],
    repr_b10: dict[str, np.ndarray],
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
        n_feat = len(feat_labels)
        if n_tok != n_feat or arr_b10.shape[1] != n_feat:
            raise RuntimeError(
                f"Expected (M,d,E) with d={n_feat} attribute columns (3.4-style); "
                f"got full tok={n_tok}, b10 tok={arr_b10.shape[1]} at stage={stage}."
            )
        feat_full = arr_full
        feat_b10 = arr_b10

        emb_full = feat_full.reshape(n_common * n_feat, emb_dim)
        emb_b10 = feat_b10.reshape(n_common * n_feat, emb_dim)

        pc_full, pca_full = _pca_flat_embeddings(emb_full, n_common, n_feat)
        pc_b10, pca_b10 = _pca_flat_embeddings(emb_b10, n_common, n_feat)
        if pc_full is None or pc_b10 is None:
            continue

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
                        "explained_var_pc1": float(pca_full.explained_variance_ratio_[0]),
                        "explained_var_pc2": float(pca_full.explained_variance_ratio_[1]),
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
                        "explained_var_pc1": float(pca_b10.explained_variance_ratio_[0]),
                        "explained_var_pc2": float(pca_b10.explained_variance_ratio_[1]),
                    }
                )
    return pd.DataFrame(rows)


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
    y_train_bin, keep = target_fixed_bins(df_train[TARGET])
    n_before = len(df_train)
    df_train = df_train.loc[keep].reset_index(drop=True)
    y_train_bin = y_train_bin[keep]
    print(
        f"Fixed target bins [0,0.3), [0.3,0.9), [0.9,1.1] on normalized {TARGET}: "
        f"kept {len(df_train)} / {n_before} rows ({N_TARGET_CLASSES} classes)."
    )

    X_train = df_train[FEATURES]
    scaler = sk.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    idx_b10 = build_b10_indices(n_train=X_train_scaled.shape[0])
    n_b10 = int(idx_b10.shape[0])
    pos_b10 = np.arange(n_b10, dtype=np.int64)
    idx_global = np.asarray(idx_b10, dtype=np.int64)
    X_train_attn = np.asarray(X_train_scaled[idx_global])
    y_train_bin_attn = y_train_bin[idx_global]

    model_full = TabPFNClassifier(
        model_path=CHECKPOINT_FILE,
        n_estimators=N_ESTIMATORS,
        ignore_pretraining_limits=True,
        random_state=ENS_SEED,
        device=TABPFN_EMBED_DEVICE,
    )
    model_full.fit(X_train_scaled, y_train_bin)

    model_b10 = TabPFNClassifier(
        model_path=CHECKPOINT_FILE,
        n_estimators=N_ESTIMATORS,
        ignore_pretraining_limits=True,
        random_state=int(ENS_SEED + (B10_MEMBER_INDEX_1_BASED - 1)),
        device=TABPFN_EMBED_DEVICE,
    )
    model_b10.fit(np.asarray(X_train_scaled[idx_b10]), np.asarray(y_train_bin[idx_b10]))

    n_layers = len(model_full.models_[0].transformer_encoder.layers)
    layers_used = [l for l in LAYERS if l <= n_layers]

    repr_full = extract_stage_embeddings_v34(
        model_full,
        X_train_attn,
        layers_used,
        FEATURES,
        train_row_indices=idx_global,
    )
    repr_b10 = extract_stage_embeddings_v34(
        model_b10,
        X_train_attn,
        layers_used,
        FEATURES,
        train_row_indices=pos_b10,
    )

    pca_rows = build_pca_rows(
        repr_full=repr_full,
        repr_b10=repr_b10,
        y_bin_by_sample=y_train_bin_attn,
    )
    pca_rows.to_csv(OUT_FEATURE_TOKEN_PCA_LONG, index=False)

    print("Classifier checkpoint embedding / PCA rows complete.")
    print(f"Wrote: {OUT_FEATURE_TOKEN_PCA_LONG}")
    print("Attention CSV: run Code/3.5.attention.py. For Fig. 3 run Code/4.3.Fig.3.R.")


if __name__ == "__main__":
    main()
