# Protein-Peptide Binding Affinity Predictor

Predicts binding affinity (pK) for a protein-peptide complex from a PDB
structure. The peptide is specified by its **chain ID**; the binding pocket is
detected automatically as the protein residues within a distance cutoff of the
peptide.

## How it works

The model is a 50/50 ensemble of two HistGradientBoostingRegressor heads:

| Head        | Input features                                   | Dim |
|-------------|--------------------------------------------------|-----|
| Structural  | ProteinMPNN encoder embeddings -> fine-tuned MLP | 64  |
| Sequence    | hand-crafted seq features (54) + MPNN scores (24)| 78  |

Final prediction = `0.5 * pred_structural + 0.5 * pred_sequence`.

Feature pipeline (all orchestrated by `src/calculate_features.py`):

```
pocket.identify_peptide_and_pocket   # chain ID -> (chain, res_id) keys
        |
feature_extraction_score             # ProteinMPNN embeddings [512] + scores [24]
        |
apply_mlp_to_embeddings              # MLP transform -> learned features [64]
        |
extract_sequence_features            # hand-crafted features [54]
        |
merge_sequence_features              # [54] + [24] -> sequence+score [78]
```

### Residue identification

Peptide and pocket residues are tracked by **(chain_id, res_id)** tuples, so a
peptide residue numbered `5` and a pocket residue numbered `5` (per-chain
numbering from 1) are never confused.

## Layout

```
predict.py                 # CLI entry point (run from here)
requirements.txt
src/
  calculate_features.py    # orchestrates all feature computation
  pocket.py                # chain-aware peptide/pocket detection
  feature_extraction_score.py
  proteinMPNN_dataset_score.py
  proteinMPNN_model_score.py
  apply_mlp_to_embeddings.py
  mlp_model_trainer.py
  mlp_config.py
  extract_sequence_features.py
  merge_sequence_features.py
  config.py                # paths to weights & trained models
  models/                  # <-- put mlp_final.pth, mpnn_full.joblib, seq_full.joblib here
  ProteinMPNN/             # <-- put protein_mpnn_utils.py + v_48_020.pt here
```

## Setup

```bash
pip install -r requirements.txt

# ProteinMPNN backbone
git clone https://github.com/dauparas/ProteinMPNN
cp ProteinMPNN/protein_mpnn_utils.py src/ProteinMPNN/
cp ProteinMPNN/vanilla_model_weights/v_48_020.pt src/ProteinMPNN/vanilla_model_weights/

# Trained artefacts
cp /path/to/mlp_final.pth   src/models/
cp /path/to/mpnn_full.joblib src/models/
cp /path/to/seq_full.joblib  src/models/
```

## Usage

Single complex:

```bash
python predict.py --pdb complex.pdb --peptide-chain C
```

Batch (CSV with columns `pdb_path`, `peptide_chain`, optional `pK`):

```bash
python predict.py --csv complexes.csv --output predictions.csv
```

Options: `--radius 5.0`, `--workdir features`, `--device cpu|cuda`,
`--no-save-features`.

Intermediate feature files are written to `--workdir` (default `features/`).
The output CSV has `id, pred_mpnn, pred_seq, pred_pK` (plus `y_true` and printed
metrics when `pK` is supplied).
