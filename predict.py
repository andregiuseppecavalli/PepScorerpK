#!/usr/bin/env python3
"""
predict.py
==========
Predict protein-peptide binding affinity (pK) for one or more complexes.

The peptide is specified by its chain ID; the binding pocket is derived
automatically as the protein residues within --radius Angstroms of the peptide.

Usage
-----
Single complex:
    python predict.py --pdb complex.pdb --peptide-chain C

Several complexes from a CSV (columns: pdb_path, peptide_chain, [pK]):
    python predict.py --csv complexes.csv

Common options:
    --radius 5.0            pocket cutoff in Angstroms
    --workdir ./features    where intermediate .npz files are written
    --output predictions.csv
    --device cpu|cuda

Outputs
-------
A predictions CSV with columns:
    id, pred_mpnn, pred_seq, pred_pK   (and y_true / metrics if pK was supplied)

Intermediate feature files (embeddings_pmpnn.npz, mpnn_scores.npz,
embeddings_pmpnn_learned.npz, sequence_features.npz, sequence_emb_combined.npz)
are written to --workdir.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Make the src/ package importable regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)

import config                       # noqa: E402
from calculate_features import calculate_features  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict protein-peptide binding affinity (pK).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb", help="Path to a single PDB file.")
    src.add_argument("--csv", help="CSV with columns: pdb_path, peptide_chain, [pK].")

    p.add_argument("--peptide-chain",
                   help="Peptide chain ID (e.g. C). Required with --pdb.")
    p.add_argument("--radius", type=float, default=5.0,
                   help="Pocket distance cutoff in Angstroms.")
    p.add_argument("--workdir", default="features",
                   help="Directory for intermediate feature .npz files.")
    p.add_argument("--output", default="predictions.csv",
                   help="Output predictions CSV.")
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                   help="Force device. Default: auto-detect.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-save-features", action="store_true",
                   help="Do not write intermediate feature .npz files.")
    return p.parse_args()


def build_inputs(args):
    """Build the list of {id, pdb_path, peptide_chain, label} dicts."""
    if args.pdb:
        if not args.peptide_chain:
            sys.exit("error: --peptide-chain is required when using --pdb.")
        if not os.path.exists(args.pdb):
            sys.exit(f"error: PDB not found: {args.pdb}")
        cid = os.path.splitext(os.path.basename(args.pdb))[0]
        return [{"id": cid, "pdb_path": args.pdb,
                 "peptide_chain": args.peptide_chain}]

    # CSV mode
    df = pd.read_csv(args.csv)
    required = {"pdb_path", "peptide_chain"}
    if not required.issubset(df.columns):
        sys.exit(f"error: CSV must contain columns {required}. "
                 f"Found: {list(df.columns)}")
    inputs = []
    for _, row in df.iterrows():
        pdb_path = str(row["pdb_path"])
        item = {
            "id": os.path.splitext(os.path.basename(pdb_path))[0],
            "pdb_path": pdb_path,
            "peptide_chain": str(row["peptide_chain"]),
        }
        if "pK" in df.columns and not pd.isna(row["pK"]):
            item["label"] = float(row["pK"])
        inputs.append(item)
    return inputs


def check_dim(model, X, name):
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and n_in != X.shape[1]:
        sys.exit(
            f"error: {name} expects {n_in} features but the computed matrix "
            f"has {X.shape[1]}. The feature pipeline and the trained model are "
            f"out of sync (check MLP architecture / sequence-feature layout)."
        )


def calc_metrics(y_true, y_pred):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)
    r = pearsonr(y_true, y_pred)[0]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rs = spearmanr(y_true, y_pred)[0]
    return mae, r, rmse, rs


def main():
    args = parse_args()

    for f in (config.MPNN_WEIGHTS, config.MLP_CHECKPOINT,
              config.MPNN_REGRESSOR, config.SEQ_REGRESSOR):
        if not os.path.exists(f):
            sys.exit(f"error: required model file missing: {f}")

    device = None if args.device is None else __import__("torch").device(args.device)
    inputs = build_inputs(args)

    print(f"Predicting for {len(inputs)} complex(es). Features -> {args.workdir}\n")
    feats = calculate_features(
        inputs, workdir=args.workdir, radius=args.radius,
        device=device, batch_size=args.batch_size,
        save_intermediates=not args.no_save_features,
    )

    # ---- Load trained HGBR heads ----
    m_mpnn = joblib.load(config.MPNN_REGRESSOR)
    m_seq  = joblib.load(config.SEQ_REGRESSOR)

    X_mpnn = feats["mpnn_learned"]
    X_seq  = feats["seq_combined"]
    check_dim(m_mpnn, X_mpnn, "MPNN regressor (mpnn_full.joblib)")
    check_dim(m_seq,  X_seq,  "sequence regressor (seq_full.joblib)")

    # ---- Predict + 50/50 ensemble ----
    pred_mpnn = m_mpnn.predict(X_mpnn)
    pred_seq  = m_seq.predict(X_seq)
    pred = config.W_MPNN * pred_mpnn + config.W_SEQ * pred_seq

    out = pd.DataFrame({
        "id":        feats["ids"],
        "pred_mpnn": pred_mpnn,
        "pred_seq":  pred_seq,
        "pred_pK":   pred,
    })

    labels = feats["labels"]
    have_labels = not np.all(np.isnan(labels))
    if have_labels:
        out["y_true"] = labels
        mask = ~np.isnan(labels)
        if mask.sum() >= 2:
            mae, r, rmse, rs = calc_metrics(labels[mask], pred[mask])
            print(f"\nEnsemble metrics (n={int(mask.sum())}): "
                  f"MAE={mae:.4f} R={r:.4f} RMSE={rmse:.4f} Rs={rs:.4f}")

    out.to_csv(args.output, index=False)
    print(f"\nWrote predictions -> {args.output}")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
