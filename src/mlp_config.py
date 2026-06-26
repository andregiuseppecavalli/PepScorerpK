"""
MLP architecture config. At inference time only the architecture keys matter
(hidden_dims, dropout, activation, batch_norm, layer_norm, extract_layer);
these MUST match the settings used to train mlp_final.pth or the state_dict
load will fail with a shape mismatch.
"""

mlp_config = {
    # ---- architecture (must match the trained checkpoint) ----
    "hidden_dims": [256, 128, 64],
    "dropout":     0.4,
    "activation":  "gelu",
    "batch_norm":  False,
    "layer_norm":  True,

    # Which hidden layer to extract as the learned embedding.
    # -1 = last hidden layer (penultimate, right before the head) -> 64-dim.
    "extract_layer": -1,
}
