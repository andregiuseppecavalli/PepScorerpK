"""
Central configuration for the deployed protein-peptide affinity predictor.

All paths are resolved relative to this file (the `src/` directory) so the
repository works regardless of the current working directory.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ProteinMPNN backbone (clone the official repo into src/ProteinMPNN) -----
# Needs:
#   src/ProteinMPNN/protein_mpnn_utils.py
#   src/ProteinMPNN/vanilla_model_weights/v_48_020.pt
MPNN_WEIGHTS = os.path.join(SRC_DIR, "ProteinMPNN", "vanilla_model_weights", "v_48_020.pt")
K_NEIGHBORS  = 48

# --- Trained downstream models (place your artefacts in src/models) ----------
MLP_CHECKPOINT = os.path.join(SRC_DIR, "models", "mlp_final.pth")   # MLP fine-tuner (.pth)
MPNN_REGRESSOR = os.path.join(SRC_DIR, "models", "mpnn_full.joblib")  # HGBR on MLP-learned feats
SEQ_REGRESSOR  = os.path.join(SRC_DIR, "models", "seq_full.joblib")   # HGBR on sequence+score feats

# --- Reproducibility ---------------------------------------------------------
# ProteinMPNN's autoregressive decoding order is randomized per call, so the
# score features have small run-to-run noise unless we fix the seed.
SEED = 16

# --- Ensemble weighting (50/50 average of the two HGBR heads) ----------------
W_MPNN = 0.5
W_SEQ  = 0.5
