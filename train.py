#!/usr/bin/env python
"""
train.py -- retrain the protein-peptide pK predictor.

Everything is controlled by src/config.py. The pipeline:

  1. Load the dataset CSV (default: data/frames_balanced_480_modpK.csv).
  2. Get the feature matrices. For each one, if its path is set in config.py
     the .npz is loaded directly; if the path is None it is computed from the
     PDB files with the same code used by predict.py (calculate_features.py):
        MLP_EMBEDDINGS_FILE   [N, 64]   structural HGB input
        MPNN_EMBEDDINGS_FILE  [N, 512]  MLP input (only needed to retrain the MLP)
        SEQUENCE_EMB_FILE     [N, 78]   sequence HGB input
  3. If MLP_EMBEDDINGS_FILE is None: fine-tune the MLP on the 512-dim
     embeddings (k-fold CV only to choose the epoch count, then a full refit)
     and extract its 64-dim last-hidden-layer features for every row.
  4. Fit the two HGB regressors on all 'Train' frames.
  5. Optionally report the 50/50 ensemble on first-frame 'Test' rows.

Usage:
    python train.py
    python train.py --device cuda --install      # also copy models into src/models/

Outputs (in config.TRAIN_OUTPUT_DIR, default training_output/):
    models/mlp_final.pth, models/mpnn_full.joblib, models/seq_full.joblib
    features/*.npz            (only the features computed in this run)
    test_predictions.csv, test_metrics.csv, config_used.py
"""
import argparse
import os
import re
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config  # noqa: E402

# The seed is fixed once, at the start, for the whole run.
SEED = config.SEED


# =============================================================================
# Helpers
# =============================================================================
def clean_id(s):
    """Same id normalisation as the original training scripts."""
    return re.sub(r"^\d+_", "", str(s))


def load_npz(path, name):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {name} is set in config.py but not found:\n  {path}\n"
                 f"Set it to None to compute this feature from the PDB files.")
    data = np.load(path, allow_pickle=True)
    X = data["embeddings"]
    print(f"  {name}: {os.path.basename(path)} -> {X.shape}")
    return data["ids"], X


def align(df_ids, ids, X):
    """Rows of X reordered to df_ids. Returns (found_mask, X_aligned_or_nan)."""
    lut = {clean_id(i): k for k, i in enumerate(ids)}
    rows = np.array([lut.get(i, -1) for i in df_ids])
    found = rows >= 0
    out = np.full((len(df_ids), X.shape[1]), np.nan, dtype=np.float32)
    out[found] = X[rows[found]]
    return found, out


def calc_metrics(y_true, y_pred):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R":    pearsonr(y_true, y_pred)[0],
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Rs":   spearmanr(y_true, y_pred)[0],
    }


def make_portable(model):
    """
    Drop the fitted RNG HistGradientBoostingRegressor keeps after fit. It is
    never used by .predict(), and its pickle is numpy-version specific
    (loading on another numpy raises 'PCG64 is not a known BitGenerator').
    """
    if hasattr(model, "_feature_subsample_rng"):
        del model._feature_subsample_rng
    return model


def compute_raw_features(df, workdir, device):
    """Build inputs from the CSV + PDB_DIR and run the model-free feature stage."""
    from calculate_features import calculate_raw_features

    has_chain_col = "peptide_chain" in df.columns
    inputs, missing = [], []
    for _, row in df.iterrows():
        fid = str(row[config.ID_COL])
        fname = config.PDB_FILENAME.format(id=fid, pdb=row.get("PDB", ""))
        path = os.path.join(config.PDB_DIR, fname)
        if not os.path.exists(path):
            missing.append(path)
            continue
        inputs.append({
            "id":            fid,
            "pdb_path":      path,
            "peptide_chain": str(row["peptide_chain"]) if has_chain_col else config.PEPTIDE_CHAIN,
            "label":         float(row[config.TARGET_COL]),
        })
    if missing:
        print(f"  warning: {len(missing)} PDB files not found (e.g. {missing[0]})")
    if not inputs:
        sys.exit(f"ERROR: no PDB files found in {config.PDB_DIR} "
                 f"(pattern '{config.PDB_FILENAME}').")
    print(f"  computing features for {len(inputs)} structures from {config.PDB_DIR}")

    raw = calculate_raw_features(inputs, workdir=workdir, radius=config.POCKET_RADIUS,
                                 device=device, batch_size=config.FEATURE_BATCH_SIZE,
                                 save_intermediates=True, verbose=False)
    return raw


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Retrain the peptide pK predictor (see src/config.py).")
    ap.add_argument("--csv", default=config.DATASET_CSV,
                    help=f"dataset CSV (default: {os.path.relpath(config.DATASET_CSV, config.REPO_DIR)})")
    ap.add_argument("--output-dir", default=config.TRAIN_OUTPUT_DIR,
                    help="where models, computed features and metrics are written")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="torch device (default: cuda if available)")
    ap.add_argument("--install", action="store_true",
                    help="also copy the trained models into src/models/ for predict.py")
    args = ap.parse_args()

    import torch
    from mlp_finetune import seed_everything, train_mlp, extract_embeddings
    from sklearn.ensemble import HistGradientBoostingRegressor

    seed_everything(SEED)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.output_dir
    models_dir = os.path.join(out_dir, "models")
    feats_dir = os.path.join(out_dir, "features")
    os.makedirs(models_dir, exist_ok=True)
    shutil.copyfile(os.path.join(config.SRC_DIR, "config.py"),
                    os.path.join(out_dir, "config_used.py"))

    print(f"Seed: {SEED} | device: {device} | output: {out_dir}")

    # ---- Dataset ----
    df = pd.read_csv(args.csv)
    print(f"Dataset: {args.csv} ({len(df)} rows)")
    df_ids = df[config.ID_COL].apply(clean_id).values

    # ---- Decide which features to load vs. compute ----
    retrain_mlp = config.MLP_EMBEDDINGS_FILE is None
    need_raw_mpnn = retrain_mlp and config.MPNN_EMBEDDINGS_FILE is None
    need_seq = config.SEQUENCE_EMB_FILE is None

    print("\n=== Features ===")
    sources = {}  # name -> (ids, X)
    if need_raw_mpnn or need_seq:
        raw = compute_raw_features(df, feats_dir, device)
        if need_raw_mpnn:
            sources["mpnn512"] = (raw["ids"], raw["mpnn_embeddings"])
        if need_seq:
            sources["seq78"] = (raw["ids"], raw["seq_combined"])
        print(f"  computed features saved to {feats_dir}")
    if retrain_mlp and not need_raw_mpnn:
        sources["mpnn512"] = load_npz(config.MPNN_EMBEDDINGS_FILE, "MPNN_EMBEDDINGS_FILE")
    if not retrain_mlp:
        sources["mlp64"] = load_npz(config.MLP_EMBEDDINGS_FILE, "MLP_EMBEDDINGS_FILE")
    if not need_seq:
        sources["seq78"] = load_npz(config.SEQUENCE_EMB_FILE, "SEQUENCE_EMB_FILE")

    # ---- Align everything to the CSV; keep rows present in every source ----
    keep = np.ones(len(df), dtype=bool)
    aligned = {}
    for name, (ids, X) in sources.items():
        found, aligned[name] = align(df_ids, ids, X)
        if (~found).any():
            print(f"  {name}: {(~found).sum()} CSV rows have no features "
                  f"(e.g. {df_ids[~found][:3].tolist()})")
        keep &= found
    df = df[keep].reset_index(drop=True)
    aligned = {k: v[keep] for k, v in aligned.items()}
    print(f"  rows with all features: {len(df)}")

    y = df[config.TARGET_COL].values.astype(np.float32)
    frame = df[config.ID_COL].astype(str)
    is_first_frame = (frame.str.contains("replica1_") & frame.str.endswith("_1")).values
    test_mask = (df[config.SPLIT_COL] == "Test").values
    train_mask = (df[config.SPLIT_COL] == "Train").values
    print(f"  train rows: {train_mask.sum()} | test rows: {test_mask.sum()} "
          f"(first-frame test: {(test_mask & is_first_frame).sum()})")

    # ---- MLP ----
    mlp_ckpt_out = os.path.join(models_dir, "mlp_final.pth")
    if retrain_mlp:
        print("\n=== MLP fine-tuning (CV to pick the epoch count, then full refit) ===")
        tv = ~test_mask
        mlp, n_ep = train_mlp(
            aligned["mpnn512"][tv], y[tv],
            groups_tv=df.loc[tv, config.GROUP_COL].values,
            is_first_frame_tv=is_first_frame[tv],
            cfg=config.MLP_CONFIG, device=device, seed=SEED,
        )
        torch.save(mlp.state_dict(), mlp_ckpt_out)
        X64 = extract_embeddings(mlp, aligned["mpnn512"],
                                 layer_idx=config.MLP_CONFIG["extract_layer"], device=device)
        os.makedirs(feats_dir, exist_ok=True)
        np.savez_compressed(os.path.join(feats_dir, "embeddings_mpnn_mlp_finetuned.npz"),
                            embeddings=X64, ids=frame.values, labels=y)
        print(f"  MLP saved ({n_ep} epochs) -> {mlp_ckpt_out}")
        print(f"  MLP features {X64.shape} -> {feats_dir}/embeddings_mpnn_mlp_finetuned.npz")
    else:
        X64 = aligned["mlp64"]
        print("\n=== MLP: using precomputed MLP_EMBEDDINGS_FILE (not retrained) ===")
        if os.path.exists(config.MLP_CHECKPOINT):
            shutil.copyfile(config.MLP_CHECKPOINT, mlp_ckpt_out)
            print(f"  copied the matching checkpoint {config.MLP_CHECKPOINT}")
        else:
            print(f"  warning: {config.MLP_CHECKPOINT} not found. predict.py needs the "
                  f"MLP checkpoint that produced MLP_EMBEDDINGS_FILE.")
    X78 = aligned["seq78"]

    # ---- HGB regressors (all Train frames) ----
    print("\n=== HGB regressors ===")
    m_mpnn = HistGradientBoostingRegressor(random_state=SEED, **config.HGB_MPNN_PARAMS)
    m_mpnn.fit(X64[train_mask], y[train_mask])
    m_seq = HistGradientBoostingRegressor(random_state=SEED, **config.HGB_SEQ_PARAMS)
    m_seq.fit(X78[train_mask], y[train_mask])
    joblib.dump(make_portable(m_mpnn), os.path.join(models_dir, "mpnn_full.joblib"))
    joblib.dump(make_portable(m_seq), os.path.join(models_dir, "seq_full.joblib"))
    print(f"  structural HGB on {X64.shape[1]} feats, sequence HGB on {X78.shape[1]} feats "
          f"-> {models_dir}")

    # ---- Test evaluation (first-frame test rows) ----
    if config.EVALUATE_TEST and (test_mask & is_first_frame).any():
        te = test_mask & is_first_frame
        p_mpnn = m_mpnn.predict(X64[te])
        p_seq = m_seq.predict(X78[te])
        p_ens = config.W_MPNN * p_mpnn + config.W_SEQ * p_seq
        print(f"\n=== Test set (first-frame rows, n={te.sum()}) ===")
        rows = []
        for name, p in (("mpnn_full", p_mpnn), ("seq_full", p_seq), ("ensemble", p_ens)):
            m = calc_metrics(y[te], p)
            rows.append({"model": name, **m})
            print(f"  [{name:>9s}] MAE={m['MAE']:.4f} | R={m['R']:.4f} | "
                  f"RMSE={m['RMSE']:.4f} | Rs={m['Rs']:.4f}")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
        pd.DataFrame({config.ID_COL: frame.values[te], "y_true": y[te],
                      "pred_mpnn": p_mpnn, "pred_seq": p_seq, "pred_pK": p_ens}
                     ).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # ---- Optionally install for predict.py ----
    if args.install:
        dst = os.path.dirname(config.MPNN_REGRESSOR)
        os.makedirs(dst, exist_ok=True)
        for f in ("mlp_final.pth", "mpnn_full.joblib", "seq_full.joblib"):
            src = os.path.join(models_dir, f)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(dst, f))
        print(f"\nModels installed into {dst} (used by predict.py).")

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
