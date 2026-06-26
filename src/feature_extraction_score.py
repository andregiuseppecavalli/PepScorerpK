"""
Compute ProteinMPNN-derived features for a set of complexes.

Returns, for each record:
    * structural embeddings  [N, 512]   (pep | pock | prod | diff)
    * peptide score features [N, 24]    (logprob, recovery, entropy, 21-dim dist)

This is the importable form of the original feature_extraction_score.py
(paths and CSV loading removed; it now consumes in-memory records).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from proteinMPNN_dataset_score import MPNNDataset, collate_mpnn
from proteinMPNN_model_score import ProteinMPNNEncoderScorer


def compute_mpnn_features(records, model_weights, device,
                          batch_size=16, num_workers=0,
                          k_neighbors=48, seed=None):
    """
    Args:
        records:        list of dicts (see MPNNDataset docstring)
        model_weights:  path to ProteinMPNN .pt checkpoint
        device:         torch.device
        seed:           optional int for reproducible decoding order

    Returns:
        ids     [N]      complex identifiers (kept rows only)
        emb     [N, 512] structural embeddings
        scores  [N, 24]  peptide score features
        labels  [N]      pK labels (NaN where unknown)
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    dataset = MPNNDataset(records)
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_mpnn,
        num_workers=num_workers, shuffle=False,
    )

    model = ProteinMPNNEncoderScorer(model_weights, device, k_neighbors=k_neighbors)

    all_emb, all_scores, all_lbl, all_ids = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="MPNN features"):
            if batch is None:
                continue
            X, chain_M, res_idx, mask, m_pep, m_pock, S, lbl, ids = batch

            X       = X.to(device)
            chain_M = chain_M.to(device)
            res_idx = res_idx.to(device)
            mask    = mask.to(device)
            m_pep   = m_pep.to(device)
            m_pock  = m_pock.to(device)
            S       = S.to(device)

            out = model(X, mask, chain_M, res_idx, S, m_pep, m_pock)
            all_emb.append(out["embeddings"].cpu().numpy())
            all_scores.append(out["score_features"].cpu().numpy())
            all_lbl.append(lbl.numpy())
            all_ids.extend(ids)

    if not all_ids:
        raise RuntimeError(
            "No complexes were successfully featurized. Check the warnings "
            "above (peptide/pocket masks empty, unreadable PDB, etc.)."
        )

    emb    = np.concatenate(all_emb, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_lbl, axis=0)
    ids    = np.array(all_ids)
    return ids, emb, scores, labels


score_feature_names = ProteinMPNNEncoderScorer.score_feature_names
