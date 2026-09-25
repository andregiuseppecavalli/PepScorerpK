"""
mlp_finetune.py
---------------
Fine-tune the MLP head on raw ProteinMPNN embeddings [512] and extract the
last-hidden-layer activations [64] used by the structural HGB regressor.

Importable form of the original main_MLP_finetune.py:
  * k-fold StratifiedGroupKFold CV on train+val, used ONLY to find the best
    epoch in each fold (no CV metrics / predictions are saved);
  * refit on the full train+val for round(mean best epoch) epochs;
  * extraction of the hidden-layer activations for any set of rows.
"""
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from mlp_model_trainer import MLP, MLPTrainer, make_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_folds(df, groups, n_s=5, n_grp=10, target_col="pK"):
    df = df.copy()
    df["Fold"] = -1
    skf = StratifiedGroupKFold(n_splits=n_s)
    df["grp"] = pd.cut(df[target_col], n_grp, labels=False)
    target = df.grp
    for fold_no, (_, v) in enumerate(skf.split(target, target, groups=groups)):
        df.loc[v, "Fold"] = fold_no
    return df


def make_loader(X, y, batch_size, shuffle, generator=None):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())

    def collate(rows):
        return {"x": torch.stack([r[0] for r in rows]),
                "y": torch.stack([r[1] for r in rows])}

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=generator, num_workers=0)


def build_model(input_dim, cfg, device):
    return MLP(
        input_dim=input_dim,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],
        batch_norm=cfg["batch_norm"],
        layer_norm=cfg["layer_norm"],
    ).to(device)


def build_scheduler(cfg, optimizer, n_epochs, steps_per_epoch):
    if cfg["scheduler"] == "cycle":
        return OneCycleLR(optimizer, max_lr=cfg["lr"], epochs=n_epochs,
                          steps_per_epoch=steps_per_epoch, pct_start=0.1,
                          div_factor=cfg["div_factor"],
                          final_div_factor=cfg["final_div_factor"])
    if cfg["scheduler"] == "exp":
        return ExponentialLR(optimizer, cfg["gamma"])
    return None


def _fit(X_tr, y_tr, X_va, y_va, n_epochs, patience, cfg, device, generator):
    train_dl = make_loader(X_tr, y_tr, cfg["batch_size"], shuffle=True, generator=generator)
    val_dl   = make_loader(X_va, y_va, cfg["batch_size"], shuffle=False, generator=generator)
    model = build_model(X_tr.shape[1], cfg, device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(cfg, optimizer, n_epochs, max(len(train_dl), 1))
    trainer = MLPTrainer(model, device, make_loss(cfg["loss"]), optimizer, scheduler)
    _, _, best_epoch = trainer.train(n_epochs=n_epochs, train_dl=train_dl,
                                     val_dl=val_dl, patience=patience)
    return model, best_epoch


def train_mlp(X_tv, y_tv, groups_tv, is_first_frame_tv, cfg, device, seed):
    """
    Args:
        X_tv              [N, 512] raw MPNN embeddings of the train+val rows
        y_tv              [N]      pK
        groups_tv         [N]      group labels for StratifiedGroupKFold
        is_first_frame_tv [N]      bool, rows used for validation scoring
        cfg               config.MLP_CONFIG
    Returns:
        final_model (eval mode), n_epochs_full
    """
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    X_tv = X_tv.astype(np.float32)
    y_tv = y_tv.astype(np.float32)
    use_all = cfg["use_all_frames"]
    n_folds = cfg["n_folds"]

    folds_df = create_folds(pd.DataFrame({"pK": y_tv}), groups_tv,
                            n_s=n_folds, n_grp=cfg["n_strat_bins"], target_col="pK")
    fold_ids = folds_df["Fold"].values

    # ---- CV: only to find the best epoch of each fold ----
    best_epochs = []
    for i in range(n_folds):
        val_all = fold_ids == i
        tr_mask = ~val_all if use_all else (~val_all & is_first_frame_tv)
        va_mask = val_all & is_first_frame_tv
        print(f"\n  ----- MLP CV fold {i+1}/{n_folds}: "
              f"{tr_mask.sum()} train rows, {va_mask.sum()} val rows (first-frame) -----")
        _, best_epoch = _fit(X_tv[tr_mask], y_tv[tr_mask], X_tv[va_mask], y_tv[va_mask],
                             cfg["n_epochs"], cfg["patience"], cfg, device, g)
        best_epochs.append(best_epoch)

    n_epochs_full = max(int(round(float(np.mean(best_epochs)))), 1)
    print(f"\n  best epochs per fold: {best_epochs} -> refit for {n_epochs_full} epochs")

    # ---- Refit on the full train+val ----
    if use_all:
        X_full, y_full = X_tv, y_tv
    else:
        X_full, y_full = X_tv[is_first_frame_tv], y_tv[is_first_frame_tv]
    # The first-frame train+val rows act as a monitoring set only (in-sample);
    # no early stopping is applied in the refit (patience=None).
    final_model, _ = _fit(X_full, y_full, X_tv[is_first_frame_tv], y_tv[is_first_frame_tv],
                          n_epochs_full, None, cfg, device, g)
    final_model.eval()
    return final_model, n_epochs_full


def extract_embeddings(model, X, layer_idx=-1, device=None, batch_size=1024):
    """Hidden-layer activations of `model` for every row of X."""
    device = device or next(model.parameters()).device
    X = np.asarray(X, dtype=np.float32)
    out = []
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(X), batch_size):
            chunk = torch.from_numpy(X[s:s + batch_size]).to(device)
            out.append(model.extract_features(chunk, layer_idx=layer_idx).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
