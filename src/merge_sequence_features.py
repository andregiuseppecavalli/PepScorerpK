"""
merge_sequence_features.py
--------------------------
Combine the hand-crafted sequence features (54-dim, from
extract_sequence_features.py) with the ProteinMPNN score features (24-dim, from
feature_extraction_score.py) into the 78-dim matrix that the sequence HGBR head
(seq_full.joblib) consumes.

In the original training workflow this was a standalone step run after
extract_sequence_features.py. Here it is an importable function that
calculate_features.py calls, but it can still be used on its own:

    ids, X78, _ = merge_sequence_features(seq54, seq_ids, scores24, score_ids)
"""
import numpy as np


def _align(ids_ref, ids_other, X_other):
    """
    Reorder rows of X_other so its ids line up with ids_ref.

    Returns (keep_mask, X_aligned) where keep_mask is a boolean array over
    ids_ref marking the entries that had a match in ids_other, and X_aligned
    holds only the matched rows, in ids_ref order.
    """
    idx = {str(i): k for k, i in enumerate(ids_other)}
    keep = np.array([str(i) in idx for i in ids_ref])
    rows = [idx[str(i)] for i in ids_ref if str(i) in idx]
    return keep, X_other[rows]


def merge_sequence_features(seq_feats, seq_ids, scores, score_ids):
    """
    Concatenate sequence features with MPNN score features, aligned by id.

    The MPNN score ids are used as the reference order, so the result lines up
    with the structural (MPNN-learned) features produced in the same pass.

    Args:
        seq_feats (np.ndarray): [Ns, 54] hand-crafted sequence features.
        seq_ids   (np.ndarray): [Ns]     ids for seq_feats.
        scores    (np.ndarray): [Nm, 24] MPNN score features.
        score_ids (np.ndarray): [Nm]     ids for scores (reference order).

    Returns:
        ids_common (np.ndarray): [N]      ids present in both, in score_ids order.
        combined   (np.ndarray): [N, 78]  concatenation [sequence(54) | scores(24)].
        keep       (np.ndarray): bool mask over score_ids of which rows survived.
    """
    keep, seq_aligned = _align(score_ids, seq_ids, seq_feats)
    ids_common = np.asarray(score_ids)[keep]
    scores_c = np.asarray(scores)[keep]
    combined = np.concatenate([seq_aligned, scores_c], axis=1)  # [N, 54+24 = 78]
    return ids_common, combined, keep
