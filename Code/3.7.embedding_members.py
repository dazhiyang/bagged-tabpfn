#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
#################################################################################
# This code is co-authored by:
# - Dazhi Yang (yangdazhi.nus@gmail.com)
#   School of Electrical Engineering and Automation,
#   Harbin Institute of Technology (HIT)
# - Yun Chen (PowerPuffYun) (chenyunpku@163.com)
#   Public Meteorological Service Center,
#   China Meteorological Administration (CMA)
#################################################################################

3.7.embedding_members.py — Same stage-embedding + PCA pipeline as Code/3.5.embedding.py,
but for selected TabPFN-B bootstrap members (B1–B10) only (no full-context by default).

Writes a long CSV with the same columns as ``feature_token_pca_layers_long.csv``:
``context, stage, sample_id, token, y_bin, pc1, pc2, explained_var_pc1, explained_var_pc2``.
Member contexts are labelled ``b1``…``b10``. For each member, PCA is fit on that
member's bootstrap tokens only (same as the B10 path in ``3.5``). Stages are Input
plus ``LAYERS`` (default L1, L2, L3, L6, L9, L12). Full-context PCA remains in
``3.5`` / Fig. 3(c).

Bootstrap draws and ``random_state`` match Code/2.5.TabPFN-B.py / 3.4 / 3.5 / 3.6 for
``COMBO`` (default ``yHxP``).

Select members via the parameter block ``MEMBERS`` and/or CLI / env:

  /opt/anaconda3/bin/python Code/3.7.embedding_members.py --members 1,3
  TABPFN_EMBED_MEMBERS=1,3 /opt/anaconda3/bin/python Code/3.7.embedding_members.py

Optional ``--include-full``: also fit Full on all N, PCA on all N×d tokens, and write
``context=full`` rows for every training sample (large CSV). Default device:
``TABPFN_EMBED_DEVICE`` (fallback ``cpu``, same as ``3.5``; TabPFN may reject MPS).
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import sklearn as sk
import torch
from tabpfn import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

#################################################################################
# Parameter block
#################################################################################

PROJECT_PATH = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
INPUT_FILE = os.path.join(PROJECT_PATH, "Data", "arranged15min.txt")
DIAG_DIR = os.path.join(PROJECT_PATH, "Data", "Output", "Diag")
CHECKPOINT_FILE = os.path.join(PROJECT_PATH, "tabpfn-v2-classifier-gn2p4bpt.ckpt")
OUT_FEATURE_TOKEN_PCA_LONG = os.path.join(
    DIAG_DIR, "feature_token_pca_layers_members_long.csv"
)

TRAIN_YEAR = 2024
TARGET = "yH"
FEATURES = ["xP", "SZA", "lcc", "mcc", "tcsw", "tcwv"]
COMBO = "yHxP"

N_TARGET_CLASSES = 3
N_ESTIMATORS = 1
LAYERS = [1, 2, 3, 6, 9, 12]

ENS_K = 10
ENS_SEED = 123
BOOTSTRAP_N = 2000
COMBO_ORDER = ["yHxP", "yHxS", "yLxP", "yLxS"]

# Bootstrap members only (1-based). Full context is already in 3.5 / Fig. 3(c).
MEMBERS = list(range(1, ENS_K + 1))  # B1–B10
INCLUDE_FULL = False

TABPFN_EMBED_DEVICE = os.environ.get("TABPFN_EMBED_DEVICE", "cpu")


#################################################################################
# Helpers (aligned with 3.5 / 3.6)
#################################################################################


def target_fixed_bins(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
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


def build_member_indices(n_train: int, member_1_based: int) -> np.ndarray:
    """Reproduce the bootstrap draw for member ``member_1_based`` on ``COMBO``.

    Walks the same RNG stream as Code/2.5.TabPFN-B.py / 3.4 / 3.5 / 3.6:
    for k in 0..ENS_K-1, for each combo in COMBO_ORDER, one draw of size BOOTSTRAP_N.
    """
    if member_1_based < 1 or member_1_based > ENS_K:
        raise ValueError(f"member must be in 1..{ENS_K}; got {member_1_based}")
    rng = np.random.default_rng(ENS_SEED)
    idx = None
    for k in range(ENS_K):
        for combo_name in COMBO_ORDER:
            draw = rng.integers(low=0, high=n_train, size=BOOTSTRAP_N)
            if (k == member_1_based - 1) and (combo_name == COMBO):
                idx = draw
    if idx is None:
        raise RuntimeError(f"Failed to derive B{member_1_based} sampling indices.")
    return idx


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

    Same extraction as Code/3.5.embedding.py.
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


def _pca_fit_transform_grid(
    arr: np.ndarray,
) -> tuple[np.ndarray | None, sk.decomposition.PCA | None]:
    """``arr`` shape ``(N, n_feat, emb_dim)`` → PCA grid ``(N, n_feat, 2)`` and fitted PCA."""
    if arr.ndim != 3:
        raise RuntimeError(f"Expected (N, n_feat, emb_dim); got {arr.shape}")
    n0, n_feat, _emb_dim = arr.shape
    emb = arr.reshape(n0 * n_feat, arr.shape[2])
    valid = np.isfinite(emb).all(axis=1)
    if valid.sum() < 2:
        return None, None
    x = emb[valid].astype(np.float64, copy=False)
    x = sk.preprocessing.StandardScaler().fit_transform(x)
    pca = sk.decomposition.PCA(n_components=2, random_state=ENS_SEED)
    pcs = pca.fit_transform(x)
    flat = np.full((n0 * n_feat, 2), np.nan, dtype=float)
    flat[valid, :] = pcs
    return flat.reshape(n0, n_feat, 2), pca


def build_pca_rows_one_context(
    stage_to_repr: dict[str, np.ndarray],
    y_bin_by_sample: np.ndarray,
    context: str,
    layers_1_based: list[int],
) -> pd.DataFrame:
    """PCA fit on this context's tokens only; one CSV row per (sample, token, stage)."""
    stage_order = ["Input"] + [f"L{x}" for x in layers_1_based]
    rows: list[dict] = []
    n_plot = int(len(y_bin_by_sample))
    feat_labels = FEATURES
    n_feat = len(feat_labels)

    for stage in stage_order:
        if stage not in stage_to_repr:
            continue
        arr = stage_to_repr[stage]
        if arr.shape[0] != n_plot:
            raise RuntimeError(
                f"Repr rows {arr.shape[0]} must match y_bin length {n_plot} at stage={stage}."
            )
        if arr.shape[1] != n_feat:
            raise RuntimeError(
                f"Expected (M,d,E) with d={n_feat}; got tok={arr.shape[1]} at stage={stage}."
            )

        pc_grid, pca = _pca_fit_transform_grid(arr)
        if pc_grid is None or pca is None:
            continue

        for i in range(n_plot):
            yb = int(y_bin_by_sample[i])
            for f in range(n_feat):
                rows.append(
                    {
                        "context": context,
                        "stage": stage,
                        "sample_id": int(i),
                        "token": feat_labels[f],
                        "y_bin": f"C{yb + 1}",
                        "pc1": float(pc_grid[i, f, 0]),
                        "pc2": float(pc_grid[i, f, 1]),
                        "explained_var_pc1": float(pca.explained_variance_ratio_[0]),
                        "explained_var_pc2": float(pca.explained_variance_ratio_[1]),
                    }
                )
    return pd.DataFrame(rows)


def parse_members(s: str) -> list[int]:
    parts = [p.strip() for p in re.split(r"[,\s]+", s.strip()) if p.strip()]
    out: list[int] = []
    for p in parts:
        m = re.fullmatch(r"[Bb]?(\d+)", p)
        if not m:
            raise ValueError(f"Bad member token {p!r}; use e.g. 1,5,8 or B1,B5,B8")
        k = int(m.group(1))
        if k < 1 or k > ENS_K:
            raise ValueError(f"Member {k} out of range 1..{ENS_K}")
        out.append(k)
    if not out:
        raise ValueError("No members specified")
    seen: set[int] = set()
    uniq: list[int] = []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TabPFN stage embeddings / PCA for selected B1–B10 members.")
    p.add_argument(
        "--members",
        type=str,
        default=os.environ.get(
            "TABPFN_EMBED_MEMBERS",
            ",".join(str(m) for m in MEMBERS),
        ),
        help="Comma-separated 1-based members, e.g. 1,5,8 (env: TABPFN_EMBED_MEMBERS)",
    )
    p.add_argument(
        "--include-full",
        action="store_true",
        default=None,
        help="Also extract full-context PCA (default: INCLUDE_FULL in parameter block)",
    )
    p.add_argument(
        "--no-full",
        action="store_true",
        help="Skip full-context extraction",
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.environ.get("TABPFN_EMBED_MEMBERS_OUT", OUT_FEATURE_TOKEN_PCA_LONG),
        help="Output CSV path",
    )
    return p.parse_args()


#################################################################################
# Main
#################################################################################


def main() -> None:
    args = parse_args()
    members = parse_members(args.members)
    if args.no_full:
        include_full = False
    elif args.include_full:
        include_full = True
    else:
        include_full = bool(INCLUDE_FULL)
        env_full = os.environ.get("TABPFN_EMBED_INCLUDE_FULL")
        if env_full is not None:
            include_full = env_full.strip() not in ("0", "false", "False", "")

    out_path = args.out

    if not os.path.isfile(CHECKPOINT_FILE):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}\n"
            "Place the TabPFN classifier checkpoint at this path or set CHECKPOINT_FILE."
        )

    os.makedirs(os.path.dirname(out_path) or DIAG_DIR, exist_ok=True)

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
    X_full = np.asarray(X_train_scaled)

    frames: list[pd.DataFrame] = []
    layers_used: list[int] | None = None

    if include_full:
        print(f"Fitting full context (n={X_full.shape[0]}) on device={TABPFN_EMBED_DEVICE} …")
        model_full = TabPFNClassifier(
            model_path=CHECKPOINT_FILE,
            n_estimators=N_ESTIMATORS,
            ignore_pretraining_limits=True,
            random_state=ENS_SEED,
            device=TABPFN_EMBED_DEVICE,
        )
        model_full.fit(X_train_scaled, y_train_bin)
        n_layers = len(model_full.models_[0].transformer_encoder.layers)
        layers_used = [l for l in LAYERS if l <= n_layers]
        print(f"Extracting embeddings / PCA: context=full, stages=Input+{layers_used}")
        repr_full = extract_stage_embeddings_v34(
            model_full,
            X_full,
            layers_used,
            FEATURES,
            train_row_indices=np.arange(X_full.shape[0], dtype=np.int64),
        )
        frames.append(
            build_pca_rows_one_context(
                stage_to_repr=repr_full,
                y_bin_by_sample=y_train_bin,
                context="full",
                layers_1_based=layers_used,
            )
        )

    for m in members:
        ctx = f"b{m}"
        idx = build_member_indices(n_train=X_full.shape[0], member_1_based=m)
        n_m = int(idx.shape[0])
        pos_m = np.arange(n_m, dtype=np.int64)
        X_m = np.asarray(X_train_scaled[idx])
        y_m = np.asarray(y_train_bin[idx])
        print(
            f"Fitting {ctx} (bootstrap_n={n_m}, seed={ENS_SEED + m - 1}) "
            f"on device={TABPFN_EMBED_DEVICE} …"
        )
        model_m = TabPFNClassifier(
            model_path=CHECKPOINT_FILE,
            n_estimators=N_ESTIMATORS,
            ignore_pretraining_limits=True,
            random_state=int(ENS_SEED + (m - 1)),
            device=TABPFN_EMBED_DEVICE,
        )
        model_m.fit(X_m, y_m)
        if layers_used is None:
            n_layers = len(model_m.models_[0].transformer_encoder.layers)
            layers_used = [l for l in LAYERS if l <= n_layers]
        print(f"Extracting embeddings / PCA: context={ctx}, stages=Input+{layers_used}")
        repr_m = extract_stage_embeddings_v34(
            model_m,
            X_m,
            layers_used,
            FEATURES,
            train_row_indices=pos_m,
        )
        frames.append(
            build_pca_rows_one_context(
                stage_to_repr=repr_m,
                y_bin_by_sample=y_m,
                context=ctx,
                layers_1_based=layers_used,
            )
        )

    out_df = pd.concat(frames, ignore_index=True)
    out_df.to_csv(out_path, index=False)
    print(
        f"Wrote: {out_path}  ({len(out_df)} rows; contexts={sorted(out_df['context'].unique())})"
    )


if __name__ == "__main__":
    main()
