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

3.6.attention_members.py — Same attention extraction as Code/3.4.attention.py, but for
selected TabPFN-B bootstrap members (B1–B10) only (no full-context / no delta).

Writes a long CSV with the same columns as ``attention_feature_layers_long.csv``:
``context, layer, from_feature, to_feature, attention``. Member contexts are labelled
``b1``…``b10``. Default layers are L3, L6, L9, L12 (SI heatmap: rows = members,
columns = layers). Full-context attention remains in ``3.4`` / Fig. 3(b).

Bootstrap draws and ``random_state`` match Code/2.5.TabPFN-B.py / 3.4 / 3.5 for
``COMBO`` (default ``yHxP``).

Select members via the parameter block ``MEMBERS`` and/or CLI / env:

  /opt/anaconda3/bin/python Code/3.6.attention_members.py --members 1,3
  TABPFN_ATTENTION_MEMBERS=1,3 /opt/anaconda3/bin/python Code/3.6.attention_members.py

Optional ``--include-full`` if you also want context=full in the same file.
Default device: ``mps`` (override with ``TABPFN_PREDICT_DEVICE`` / ``TABPFN_DEVICE``).
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

#################################################################################
# Parameter block
#################################################################################

PROJECT_PATH = "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
INPUT_FILE = os.path.join(PROJECT_PATH, "Data", "arranged15min.txt")
DIAG_DIR = os.path.join(PROJECT_PATH, "Data", "Output", "Diag")
CHECKPOINT_FILE = os.path.join(PROJECT_PATH, "tabpfn-v2-classifier-gn2p4bpt.ckpt")
OUT_FEATURE_LONG = os.path.join(DIAG_DIR, "attention_feature_layers_members_long.csv")

TRAIN_YEAR = 2024
TARGET = "yH"
FEATURES = ["xP", "SZA", "lcc", "mcc", "tcsw", "tcwv"]
COMBO = "yHxP"

N_TARGET_CLASSES = 3
N_ESTIMATORS = 1
# Layers shown in Fig. 3(b)-style heatmaps (no L1/L2 — not used in the SI member panel).
LAYERS = [3, 6, 9, 12]
ATTENTION_PREDICT_CHUNK = 512

ENS_K = 10
ENS_SEED = 123
BOOTSTRAP_N = 2000
COMBO_ORDER = ["yHxP", "yHxS", "yLxP", "yLxS"]

# Bootstrap members only (1-based). Full context is already in 3.4 / Fig. 3(b).
MEMBERS = list(range(1, ENS_K + 1))  # B1–B10
INCLUDE_FULL = False

TABPFN_PREDICT_DEVICE = os.environ.get(
    "TABPFN_PREDICT_DEVICE",
    os.environ.get("TABPFN_DEVICE", "mps"),
)


#################################################################################
# Helpers (aligned with 3.4)
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

    Walks the same RNG stream as Code/2.5.TabPFN-B.py / 3.4 / 3.5:
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
    # preserve order, drop duplicates
    seen: set[int] = set()
    uniq: list[int] = []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TabPFN attention for selected B1–B10 members.")
    p.add_argument(
        "--members",
        type=str,
        default=os.environ.get(
            "TABPFN_ATTENTION_MEMBERS",
            ",".join(str(m) for m in MEMBERS),
        ),
        help="Comma-separated 1-based members, e.g. 1,5,8 (env: TABPFN_ATTENTION_MEMBERS)",
    )
    p.add_argument(
        "--include-full",
        action="store_true",
        default=None,
        help="Also extract full-context attention (default: INCLUDE_FULL in parameter block)",
    )
    p.add_argument(
        "--no-full",
        action="store_true",
        help="Skip full-context extraction",
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.environ.get("TABPFN_ATTENTION_MEMBERS_OUT", OUT_FEATURE_LONG),
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
        env_full = os.environ.get("TABPFN_ATTENTION_INCLUDE_FULL")
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

    rows: list[dict] = []
    layers_used: list[int] | None = None

    if include_full:
        print(f"Fitting full context (n={X_full.shape[0]}) on device={TABPFN_PREDICT_DEVICE} …")
        model_full = TabPFNClassifier(
            model_path=CHECKPOINT_FILE,
            n_estimators=N_ESTIMATORS,
            ignore_pretraining_limits=True,
            random_state=ENS_SEED,
            device=TABPFN_PREDICT_DEVICE,
        )
        model_full.fit(X_train_scaled, y_train_bin)
        n_layers = len(model_full.models_[0].transformer_encoder.layers)
        layers_used = [l for l in LAYERS if l <= n_layers]
        print(f"Extracting attention: context=full, layers={layers_used}")
        rows.extend(extract_attention(model_full, X_full, "full", layers_used))

    for m in members:
        ctx = f"b{m}"
        idx = build_member_indices(n_train=X_full.shape[0], member_1_based=m)
        X_m = np.asarray(X_train_scaled[idx])
        y_m = np.asarray(y_train_bin[idx])
        print(
            f"Fitting {ctx} (bootstrap_n={len(idx)}, seed={ENS_SEED + m - 1}) "
            f"on device={TABPFN_PREDICT_DEVICE} …"
        )
        model_m = TabPFNClassifier(
            model_path=CHECKPOINT_FILE,
            n_estimators=N_ESTIMATORS,
            ignore_pretraining_limits=True,
            random_state=int(ENS_SEED + (m - 1)),
            device=TABPFN_PREDICT_DEVICE,
        )
        model_m.fit(X_m, y_m)
        if layers_used is None:
            n_layers = len(model_m.models_[0].transformer_encoder.layers)
            layers_used = [l for l in LAYERS if l <= n_layers]
        print(f"Extracting attention: context={ctx}, layers={layers_used}")
        rows.extend(extract_attention(model_m, X_m, ctx, layers_used))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  ({len(out_df)} rows; contexts={sorted(out_df['context'].unique())})")


if __name__ == "__main__":
    main()
