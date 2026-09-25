# data/

| Path | Content | Needed when |
|---|---|---|
| `Dataset.csv` | Dataset: `Frame`, `PDB`, `pK`, `Split` (Train/Test), `cluster alg train` (CV groups) | always (train.py) |
| `features/embeddings_mpnn_mlp.npz` | MLP-learned structural features [N, 64] | `MLP_EMBEDDINGS_FILE` is set |
| `features/embeddings_pmpnn.npz` | raw ProteinMPNN embeddings [N, 512] | the MLP is retrained from precomputed embeddings |
| `features/embeddings_seq.npz` | sequence (54) + MPNN score (24) features [N, 78] | `SEQUENCE_EMB_FILE` is set |
| `pdbs/<Frame>.pdb` | one structure per CSV row | any feature path is `None` (full replication) |

Each `.npz` contains `ids` (matching the CSV `Frame` column; a leading
`<digits>_` prefix is ignored) and `embeddings`. File names and the PDB naming
pattern are set in `src/config.py`.
