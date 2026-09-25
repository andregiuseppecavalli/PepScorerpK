import os

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

SEED = 16

MPNN_WEIGHTS = os.path.join(SRC_DIR, "ProteinMPNN", "vanilla_model_weights", "v_48_020.pt")
K_NEIGHBORS  = 48

# Trained models used by predict.py
MLP_CHECKPOINT = os.path.join(SRC_DIR, "models", "mlp.pth")     # MLP fine-tuner
MPNN_REGRESSOR = os.path.join(SRC_DIR, "models", "str_model.joblib")  # HGB on MLP feats [64]
SEQ_REGRESSOR  = os.path.join(SRC_DIR, "models", "seq_model.joblib")   # HGB on seq+score [78]

# Ensemble weighting (50/50 average of the two HGB heads)
W_MPNN = 0.5
W_SEQ  = 0.5

# MLP fine-tuner
MLP_CONFIG = {
    "hidden_dims":   [256, 128, 64],
    "dropout":       0.4,
    "activation":    "gelu",       # 'relu', 'gelu', 'silu'
    "batch_norm":    False,
    "layer_norm":    True,
    "extract_layer": -1,        
    "n_folds":        5,           
    "n_strat_bins":   10,          
    "use_all_frames": True,     
    "batch_size":     256,
    "n_epochs":       50,
    "patience":       8,          
    "loss":           "huber",     # 'huber' or 'mse'
    "lr":             1e-3,
    "weight_decay":   1e-2,
    "scheduler":      "cycle",     # 'cycle' (OneCycleLR), 'exp' (ExponentialLR), None
    "div_factor":     25.0,
    "final_div_factor": 1000.0,
    "gamma":          0.95,
}

# Training (train.py only)
DATASET_CSV = os.path.join(DATA_DIR, "Dataset.csv")
ID_COL      = "Frame"           
TARGET_COL  = "pK"
SPLIT_COL   = "Split"              
GROUP_COL   = "cluster alg train"
PDB_DIR        = os.path.join(DATA_DIR, "pdbs")
PDB_FILENAME   = "{id}.pdb"
PEPTIDE_CHAIN  = "C"    
POCKET_RADIUS  = 5.0   
FEATURE_BATCH_SIZE = 16 

# Precomputed feature files
# Each is an .npz with 'ids' and 'embeddings'.
# Everything set to None = full replication from the PDB structures.
FEATURES_DIR         = os.path.join(DATA_DIR, "features")
MLP_EMBEDDINGS_FILE  = os.path.join(FEATURES_DIR, "embeddings_mpnn_mlp.npz")
MPNN_EMBEDDINGS_FILE = os.path.join(FEATURES_DIR, "embeddings_pmpnn.npz")
SEQUENCE_EMB_FILE    = os.path.join(FEATURES_DIR, "embeddings_seq.npz")

# HGB regressors
HGB_MPNN_PARAMS = {"max_iter": 10, "early_stopping": False}
HGB_SEQ_PARAMS  = {"max_iter": 100, "max_leaf_nodes": 31,  "learning_rate": 0.1,  "early_stopping": False}

# Output
TRAIN_OUTPUT_DIR = os.path.join(REPO_DIR, "training_output")
EVALUATE_TEST    = True  
