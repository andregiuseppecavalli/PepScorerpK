# Trained model artefacts

Place the following files here:

- `mlp_final.pth`    -- fine-tuned MLP checkpoint (state_dict). Its architecture
                        must match `src/mlp_config.py` (hidden_dims=[256,128,64],
                        dropout=0.4, gelu, layer_norm). Extracting layer -1 gives
                        the 64-dim learned structural features.
- `mpnn_full.joblib` -- HistGradientBoostingRegressor trained on the 64-dim
                        MLP-learned features.
- `seq_full.joblib`  -- HistGradientBoostingRegressor trained on the 78-dim
                        sequence+score features (54 hand-crafted + 24 MPNN scores).

`predict.py` checks each regressor's `n_features_in_` against the computed
feature width and aborts with a clear message if they disagree.
