"""
config.py -- the single configuration file for the repository.

It controls both:
    * predict.py  (inference with the trained models)
    * train.py    (retraining the MLP and the two HGB regressors)

All paths are resolved relative to the repository, so everything works
regardless of the current working directory.
"""
import os

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

# =============================================================================
# Reproducibility -- one seed for the whole pipeline
# =============================================================================
# Used for: ProteinMPNN decoding order (score features), MLP initialisation /
# shuffling, and the HGB regressors' random_state.
SEED = 16

# =============================================================================
# ProteinMPNN backbone (from the official repo, copied into src/ProteinMPNN)
# =============================================================================
MPNN_WEIGHTS = os.path.join(SRC_DIR, "ProteinMPNN", "vanilla_model_weights", "v_48_020.pt")
K_NEIGHBORS  = 48

# =============================================================================
# Trained models used by predict.py
# =============================================================================
MLP_CHECKPOINT = os.path.join(SRC_DIR, "models", "mlp_final.pth")     # MLP fine-tuner
MPNN_REGRESSOR = os.path.join(SRC_DIR, "models", "mpnn_full.joblib")  # HGB on MLP feats [64]
SEQ_REGRESSOR  = os.path.join(SRC_DIR, "models", "seq_full.joblib")   # HGB on seq+score [78]

# Ensemble weighting (50/50 average of the two HGB heads)
W_MPNN = 0.5
W_SEQ  = 0.5

# =============================================================================
# MLP fine-tuner
# =============================================================================
# Architecture keys are used by BOTH predict.py and train.py and must match the
# checkpoint in MLP_CHECKPOINT. Training keys are used only by train.py.
MLP_CONFIG = {
    # ---- architecture ----
    "hidden_dims":   [256, 128, 64],
    "dropout":       0.4,
    "activation":    "gelu",       # 'relu', 'gelu', 'silu'
    "batch_norm":    False,
    "layer_norm":    True,
    "extract_layer": -1,           # -1 = last hidden layer -> 64-dim features

    # ---- training (train.py only) ----
    "n_folds":        5,           # CV folds, used ONLY to pick the refit epoch count
    "n_strat_bins":   10,          # pK bins for StratifiedGroupKFold
    "use_all_frames": True,        # train on all frames (val always first-frame only)
    "batch_size":     256,
    "n_epochs":       50,
    "patience":       8,           # early-stopping patience inside each CV fold
    "loss":           "huber",     # 'huber' or 'mse'
    "lr":             1e-3,
    "weight_decay":   1e-2,
    "scheduler":      "cycle",     # 'cycle' (OneCycleLR), 'exp' (ExponentialLR), None
    "div_factor":     25.0,
    "final_div_factor": 1000.0,
    "gamma":          0.95,
}

# =============================================================================
# Training (train.py only)
# =============================================================================
# ---- Dataset ----------------------------------------------------------------
DATASET_CSV = os.path.join(DATA_DIR, "frames_balanced_480_modpK.csv")
ID_COL      = "Frame"               # frame id, also the PDB file stem
TARGET_COL  = "pK"
SPLIT_COL   = "Split"               # 'Train' / 'Test'
GROUP_COL   = "cluster alg train"   # groups for the MLP CV folds (avoid leakage)

# ---- Structures (only needed when features are computed from scratch) ------
# The PDB of each row is PDB_DIR / PDB_FILENAME, where {id} is the ID_COL
# value and {pdb} the 'PDB' column, e.g. "{id}.pdb" -> replica1_1AQC_1.pdb
PDB_DIR        = os.path.join(DATA_DIR, "pdbs")
PDB_FILENAME   = "{id}.pdb"
PEPTIDE_CHAIN  = "C"     # overridden per row if the CSV has a 'peptide_chain' column
POCKET_RADIUS  = 5.0     # Angstrom
FEATURE_BATCH_SIZE = 16  # ProteinMPNN batch size

# ---- Precomputed feature files ----------------------------------------------
# Each is an .npz with 'ids' and 'embeddings'. Set a path to use the file
# directly; set it to None to compute that feature from the PDB files.
#
#   MLP_EMBEDDINGS_FILE   [N, 64]  MLP-learned structural features -> structural HGB.
#                                  If set, the MLP is NOT retrained (the existing
#                                  MLP_CHECKPOINT must be the one that produced it).
#   MPNN_EMBEDDINGS_FILE  [N, 512] raw ProteinMPNN embeddings -> MLP training.
#                                  Only used when MLP_EMBEDDINGS_FILE is None.
#   SEQUENCE_EMB_FILE     [N, 78]  sequence (54) + MPNN score (24) features -> sequence HGB.
#
# Everything set to None = full replication from the PDB structures.
FEATURES_DIR         = os.path.join(DATA_DIR, "features")
MLP_EMBEDDINGS_FILE  = os.path.join(FEATURES_DIR, "embeddings_mpnn_mlp_finetuned_last64.npz")
MPNN_EMBEDDINGS_FILE = os.path.join(FEATURES_DIR, "embeddings_pmpnn_48_multi.npz")
SEQUENCE_EMB_FILE    = os.path.join(FEATURES_DIR, "sequence_emb_combined.npz")

# ---- HGB regressors -----------------------------------------------------------
HGB_MPNN_PARAMS = {"max_iter": 10,  "max_leaf_nodes": 100, "learning_rate": 0.01, "early_stopping": False}
HGB_SEQ_PARAMS  = {"max_iter": 100, "max_leaf_nodes": 31,  "learning_rate": 0.1,  "early_stopping": False}

# ---- Output -------------------------------------------------------------------
TRAIN_OUTPUT_DIR = os.path.join(REPO_DIR, "training_output")
EVALUATE_TEST    = True   # report ensemble metrics on first-frame test rows
