"""
Transform raw ProteinMPNN embeddings into MLP-learned representations.

Loads the trained MLP checkpoint and runs the raw embeddings through it in eval
mode, extracting activations from a chosen hidden layer. This is the importable
form of the original apply_mlp_to_embeddings.py.
"""
import numpy as np
import torch

from mlp_model_trainer import MLP


def transform_embeddings(X, checkpoint_path, mlp_cfg, device,
                         batch_size=1024):
    """
    Args:
        X:               np.ndarray [N, input_dim] raw MPNN embeddings (512)
        checkpoint_path: path to mlp_final.pth (state_dict)
        mlp_cfg:         dict with hidden_dims/dropout/activation/batch_norm/
                         layer_norm/extract_layer
        device:          torch.device

    Returns:
        learned: np.ndarray [N, d] activations of the chosen hidden layer
    """
    X = np.asarray(X, dtype=np.float32)
    input_dim = X.shape[1]
    extract_layer = mlp_cfg.get("extract_layer", -1)

    model = MLP(
        input_dim=input_dim,
        hidden_dims=mlp_cfg["hidden_dims"],
        dropout=mlp_cfg["dropout"],
        activation=mlp_cfg["activation"],
        batch_norm=mlp_cfg["batch_norm"],
        layer_norm=mlp_cfg["layer_norm"],
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_chunks = []
    with torch.inference_mode():
        for s in range(0, len(X), batch_size):
            chunk = torch.from_numpy(X[s:s + batch_size]).float().to(device)
            feats = model.extract_features(chunk, layer_idx=extract_layer)
            out_chunks.append(feats.detach().cpu().numpy())

    return np.concatenate(out_chunks, axis=0).astype(np.float32)
