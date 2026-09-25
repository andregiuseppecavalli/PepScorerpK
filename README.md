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

## Setup

```bash
conda env create -f environment.yml
conda activate pepscorer

## Test the installation

After setup, test the installation by running the following command:

```bash
python predict.py --pdb Example/1AQC.pdb --peptide-chain C
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

## Retraining

Everything is controlled by `src/config.py`.

```bash
python train.py                     
python train.py --device cuda --install
```

For each feature matrix, `train.py` either loads the `.npz` given in
`config.py` or, if the path is `None`, computes it from the PDB files in
`data/pdbs/`

* **Quick retrain** (default): the shipped `.npz` files are used and only the
  two HGB regressors are refit.
* **Full replication**: set all three to `None` and provide the structures as
  `data/pdbs/<Frame>.pdb` (pattern: `PDB_FILENAME`, peptide chain: `PEPTIDE_CHAIN`).
  The `pdbs` directory can be downloaded, together with the test structures, from the Zenodo repository:

The HGB regressors are fit on all `Train` frames; if `EVALUATE_TEST` is on,
the ensemble is scored on first-frame (`replica1_..._1`) `Test` rows.

Outputs go to `training_output/` (`models/`, computed `features/`,
`test_metrics.csv`, `test_predictions.csv`, `config_used.py`). `--install`
copies the three models into `src/models/` so `predict.py` uses them.

## License

This project is licensed under the MIT License — see `LICENSE`.

## Third-party notices

`src/ProteinMPNN/protein_mpnn_utils.py` and the model weights are taken from
the [ProteinMPNN repository](https://github.com/dauparas/ProteinMPNN) by
Justas Dauparas, used under the MIT License (see `src/ProteinMPNN/LICENSE`).

If you use this software, please also cite the ProteinMPNN paper:

> Dauparas, J. et al. Robust deep learning–based protein sequence design using
> ProteinMPNN. *Science* **378**, 49–56 (2022).
> https://doi.org/10.1126/science.add2187
