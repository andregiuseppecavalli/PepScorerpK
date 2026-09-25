import copy
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================================
# Model
# ============================================================================
def _make_activation(name):
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "silu": return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """
    [input_dim] -> [hidden_dims[0]] -> ... -> [hidden_dims[-1]] -> 1

    Each hidden block is Linear -> (LayerNorm | BatchNorm | none) -> Activation -> Dropout.
    The output head is a plain Linear -> 1.

    extract_features(x, layer_idx) returns activations of the hidden layer at
    index `layer_idx` (negative indices allowed, -1 = last hidden layer).
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.3,
                 activation="gelu", batch_norm=False, layer_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)

        self.hidden_blocks = nn.ModuleList()
        prev_dim = input_dim
        for h in self.hidden_dims:
            block = nn.Sequential()
            block.add_module("linear", nn.Linear(prev_dim, h))
            if batch_norm:
                block.add_module("norm", nn.BatchNorm1d(h))
            elif layer_norm:
                block.add_module("norm", nn.LayerNorm(h))
            block.add_module("act", _make_activation(activation))
            block.add_module("drop", nn.Dropout(dropout))
            self.hidden_blocks.append(block)
            prev_dim = h

        out_in_dim = prev_dim if self.hidden_dims else input_dim
        self.head = nn.Linear(out_in_dim, 1)

    def forward(self, x):
        for block in self.hidden_blocks:
            x = block(x)
        return self.head(x).squeeze(-1)

    @torch.no_grad()
    def extract_features(self, x, layer_idx=-1):
        """Return the output of the hidden block at index `layer_idx`.
        The output head is NEVER applied here."""
        if not self.hidden_blocks:
            return x
        if layer_idx < 0:
            layer_idx = len(self.hidden_blocks) + layer_idx
        if layer_idx < 0 or layer_idx >= len(self.hidden_blocks):
            raise IndexError(
                f"layer_idx out of range: {layer_idx} "
                f"(have {len(self.hidden_blocks)} hidden layers)"
            )
        h = x
        for i, block in enumerate(self.hidden_blocks):
            h = block(h)
            if i == layer_idx:
                return h
        return h

def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    res = pearsonr(y_true, y_pred)
    r = res.statistic if hasattr(res, "statistic") else res[0]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rs = spearmanr(y_true, y_pred)[0]
    return float(mae), float(r), rmse, float(rs)


def make_loss(name):
    if name == "huber": return nn.HuberLoss()
    if name == "mse":   return nn.MSELoss()
    raise ValueError(f"Unknown loss: {name}")


class MLPTrainer:
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _step(self, dataloader, train):
        self.model.train() if train else self.model.eval()
        total_loss, n_batches = 0.0, 0
        y_trues, y_preds = [], []
        ctx = torch.enable_grad() if train else torch.inference_mode()
        with ctx:
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                if train:
                    self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None and isinstance(
                        self.scheduler, torch.optim.lr_scheduler.OneCycleLR
                    ):
                        self.scheduler.step()
                total_loss += loss.detach().item()
                n_batches += 1
                y_trues.append(y.detach().cpu().numpy())
                y_preds.append(y_pred.detach().cpu().numpy())
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, np.concatenate(y_trues), np.concatenate(y_preds)

    def train(self, n_epochs, train_dl, val_dl, patience=None):
        best_val_loss, best_state, best_epoch, best_metrics = np.inf, None, -1, {}
        current_patience = patience
        history = {k: [] for k in (
            "train_loss", "val_loss", "train_mae", "val_mae",
            "train_r", "val_r", "train_rmse", "val_rmse", "train_rs", "val_rs",
        )}
        for epoch in range(n_epochs):
            tr_loss, tr_y, tr_p = self._step(train_dl, train=True)
            vl_loss, vl_y, vl_p = self._step(val_dl,   train=False)
            tr_mae, tr_r, tr_rmse, tr_rs = calc_metrics(tr_y, tr_p)
            vl_mae, vl_r, vl_rmse, vl_rs = calc_metrics(vl_y, vl_p)
            for k, v in [
                ("train_loss", tr_loss), ("val_loss", vl_loss),
                ("train_mae", tr_mae), ("val_mae", vl_mae),
                ("train_r", tr_r), ("val_r", vl_r),
                ("train_rmse", tr_rmse), ("val_rmse", vl_rmse),
                ("train_rs", tr_rs), ("val_rs", vl_rs),
            ]:
                history[k].append(v)
            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()
            print(f"  epoch {epoch+1:>3d} | train loss={tr_loss:.4f} R={tr_r:.3f} "
                  f"| val loss={vl_loss:.4f} R={vl_r:.3f} Rs={vl_rs:.3f}")
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
                best_metrics = {
                    "val_loss": vl_loss, "train_loss": tr_loss,
                    "val_mae": vl_mae, "val_r": vl_r,
                    "val_rmse": vl_rmse, "val_rs": vl_rs,
                    "train_mae": tr_mae, "train_r": tr_r,
                    "train_rmse": tr_rmse, "train_rs": tr_rs,
                }
                current_patience = patience
            elif patience is not None:
                current_patience -= 1
                if current_patience <= 0:
                    print(f"  early stopping at epoch {epoch+1} "
                          f"(best epoch was {best_epoch})")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_metrics, history, best_epoch
