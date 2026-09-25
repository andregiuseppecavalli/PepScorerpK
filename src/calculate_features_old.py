"""
calculate_features.py
---------------------
Single entry point that turns a list of (pdb, peptide_chain) inputs into the
two feature matrices the trained HGBR heads consume:

    1. pocket.identify_peptide_and_pocket   -> (chain, res_id) keys
    2. feature_extraction_score              -> MPNN embeddings [512] + scores [24]
    3. apply_mlp_to_embeddings               -> MLP-learned structural feats [64]
    4. extract_sequence_features             -> hand-crafted sequence feats [54]
    5. merge (here)                          -> sequence+score feats [54+24 = 78]

All intermediate .npz files are written to `workdir` (default: cwd). The two
final matrices are returned aligned to a common set of ids:

    mpnn_learned   [N, 64]   -> mpnn_full.joblib
    seq_combined   [N, 78]   -> seq_full.joblib
"""
import os
import numpy as np
import torch

import config
from mlp_config import mlp_config
from pocket import identify_peptide_and_pocket
from feature_extraction_score import compute_mpnn_features, score_feature_names
from apply_mlp_to_embeddings import transform_embeddings
from extract_sequence_features import compute_sequence_features, FEATURE_NAMES
from merge_sequence_features import merge_sequence_features


def build_records(inputs, radius=5.0):
    """
    inputs: list of dicts {id, pdb_path, peptide_chain, label(optional)}
    Returns records with resolved pep_keys / pock_keys.
    """
    records = []
    for item in inputs:
        pep_keys, pock_keys = identify_peptide_and_pocket(
            item["pdb_path"], item["peptide_chain"], radius=radius
        )
        print(f"  {item['id']}: chain '{item['peptide_chain']}' -> "
              f"{len(pep_keys)} peptide residues, {len(pock_keys)} pocket residues")
        records.append({
            "id":        item["id"],
            "pdb_path":  item["pdb_path"],
            "pep_keys":  pep_keys,
            "pock_keys": pock_keys,
            "label":     float(item.get("label", float("nan"))),
        })
    return records


def calculate_features(inputs, workdir=".", radius=5.0, device=None,
                       batch_size=16, save_intermediates=True):
    """
    Args:
        inputs: list of {id, pdb_path, peptide_chain, label(optional)}
        workdir: where intermediate .npz files are written

    Returns dict:
        ids            [N]
        mpnn_learned   [N, d_mlp]   (-> mpnn_full.joblib)
        seq_combined   [N, 78]      (-> seq_full.joblib)
        labels         [N]          (NaN where unknown)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(workdir, exist_ok=True)

    print("[1/5] Identifying peptide & pocket residues...")
    records = build_records(inputs, radius=radius)

    print("[2/5] Computing ProteinMPNN embeddings + score features...")
    ids_m, emb512, scores24, labels = compute_mpnn_features(
        records, config.MPNN_WEIGHTS, device,
        batch_size=batch_size, k_neighbors=config.K_NEIGHBORS, seed=config.SEED,
    )
    if save_intermediates:
        np.savez_compressed(os.path.join(workdir, "embeddings_pmpnn.npz"),
                            embeddings=emb512, ids=ids_m, labels=labels)
        np.savez_compressed(os.path.join(workdir, "mpnn_scores.npz"),
                            embeddings=scores24, ids=ids_m, labels=labels,
                            feature_names=np.array(score_feature_names()))

    print("[3/5] Transforming embeddings through the fine-tuned MLP...")
    mpnn_learned = transform_embeddings(emb512, config.MLP_CHECKPOINT, mlp_config, device)
    if save_intermediates:
        np.savez_compressed(os.path.join(workdir, "embeddings_pmpnn_learned.npz"),
                            embeddings=mpnn_learned, ids=ids_m, labels=labels)

    print("[4/5] Extracting hand-crafted sequence features...")
    ids_s, seq54, labels_s = compute_sequence_features(records)
    if save_intermediates:
        np.savez_compressed(os.path.join(workdir, "sequence_features.npz"),
                            embeddings=seq54, ids=ids_s, labels=labels_s,
                            feature_names=np.array(FEATURE_NAMES))

    print("[5/5] Merging sequence features with MPNN score features...")
    # Delegate the 54 + 24 -> 78 concatenation to merge_sequence_features,
    # aligned on the MPNN id order so it lines up with the structural head.
    ids_common, seq_combined, keep_seq = merge_sequence_features(
        seq54, ids_s, scores24, ids_m
    )
    if not keep_seq.all():
        missing = [str(i) for i, k in zip(ids_m, keep_seq) if not k]
        print(f"  warning: {len(missing)} ids lack sequence features and are "
              f"dropped: {missing[:5]}")
    mpnn_learned_c = mpnn_learned[keep_seq]
    labels_c = labels[keep_seq]
    if save_intermediates:
        np.savez_compressed(os.path.join(workdir, "sequence_emb_combined.npz"),
                            embeddings=seq_combined, ids=ids_common, labels=labels_c)

    print(f"\nFeature matrices ready for {len(ids_common)} complex(es):")
    print(f"  MPNN-learned (structural): {mpnn_learned_c.shape}")
    print(f"  sequence + score:          {seq_combined.shape}")

    return {
        "ids":          ids_common,
        "mpnn_learned": mpnn_learned_c,
        "seq_combined": seq_combined,
        "labels":       labels_c,
    }
