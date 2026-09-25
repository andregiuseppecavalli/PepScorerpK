"""
calculate_features.py
---------------------
Turns a list of (pdb, peptide_chain) inputs into feature matrices.

Two stages, so that training and inference share exactly the same code:

  calculate_raw_features()   -- needs only ProteinMPNN, no trained model
      1. pocket.identify_peptide_and_pocket   -> (chain, res_id) keys
      2. feature_extraction_score             -> MPNN embeddings [512] + scores [24]
      3. extract_sequence_features            -> hand-crafted sequence feats [54]
      4. merge_sequence_features              -> sequence+score feats [54+24 = 78]

  calculate_features()       -- inference: raw features + trained MLP
      5. apply_mlp_to_embeddings              -> MLP-learned structural feats [64]

train.py uses stage 1-4 to build its inputs and trains the MLP itself;
predict.py uses the full chain. Intermediate .npz files go to `workdir`.
"""
import os
import numpy as np
import torch

import config
from pocket import identify_peptide_and_pocket
from feature_extraction_score import compute_mpnn_features, score_feature_names
from apply_mlp_to_embeddings import transform_embeddings
from extract_sequence_features import compute_sequence_features, FEATURE_NAMES
from merge_sequence_features import merge_sequence_features


def build_records(inputs, radius=5.0, verbose=True):
    """
    inputs: list of dicts {id, pdb_path, peptide_chain, label(optional)}
    Returns records with resolved pep_keys / pock_keys. Inputs whose pocket
    detection fails are skipped with a warning.
    """
    records = []
    for item in inputs:
        try:
            pep_keys, pock_keys = identify_peptide_and_pocket(
                item["pdb_path"], item["peptide_chain"], radius=radius
            )
        except Exception as e:  # unreadable PDB, missing chain, ...
            print(f"  skip {item['id']}: pocket detection failed ({e})")
            continue
        if verbose:
            print(f"  {item['id']}: chain '{item['peptide_chain']}' -> "
                  f"{len(pep_keys)} peptide residues, {len(pock_keys)} pocket residues")
        records.append({
            "id":        item["id"],
            "pdb_path":  item["pdb_path"],
            "pep_keys":  pep_keys,
            "pock_keys": pock_keys,
            "label":     float(item.get("label", float("nan"))),
        })
    if not records:
        raise RuntimeError("Pocket detection failed for every input complex.")
    return records


def calculate_raw_features(inputs, workdir=".", radius=5.0, device=None,
                           batch_size=16, save_intermediates=True, verbose=True):
    """
    Model-free features (only ProteinMPNN is needed).

    Returns dict:
        ids              [N]
        mpnn_embeddings  [N, 512]   (-> MLP)
        seq_combined     [N, 78]    (-> seq_full.joblib)
        labels           [N]        (NaN where unknown)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(workdir, exist_ok=True)

    print("[1/5] Identifying peptide & pocket residues...")
    records = build_records(inputs, radius=radius, verbose=verbose)

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

    print("[3/5] Extracting hand-crafted sequence features...")
    ids_s, seq54, labels_s = compute_sequence_features(records)
    if save_intermediates:
        np.savez_compressed(os.path.join(workdir, "sequence_features.npz"),
                            embeddings=seq54, ids=ids_s, labels=labels_s,
                            feature_names=np.array(FEATURE_NAMES))

    print("[4/5] Merging sequence features with MPNN score features...")
    ids_common, seq_combined, keep = merge_sequence_features(seq54, ids_s, scores24, ids_m)
    if not keep.all():
        missing = [str(i) for i, k in zip(ids_m, keep) if not k]
        print(f"  warning: {len(missing)} ids lack sequence features and are "
              f"dropped: {missing[:5]}")
    emb512_c = emb512[keep]
    labels_c = labels[keep]
    if save_intermediates:
        np.savez_compressed(os.path.join(workdir, "sequence_emb_combined.npz"),
                            embeddings=seq_combined, ids=ids_common, labels=labels_c)

    return {
        "ids":             ids_common,
        "mpnn_embeddings": emb512_c,
        "seq_combined":    seq_combined,
        "labels":          labels_c,
    }


def calculate_features(inputs, workdir=".", radius=5.0, device=None,
                       batch_size=16, save_intermediates=True):
    """
    Full inference features: raw features + trained MLP transform.

    Returns dict:
        ids            [N]
        mpnn_learned   [N, 64]   (-> mpnn_full.joblib)
        seq_combined   [N, 78]   (-> seq_full.joblib)
        labels         [N]       (NaN where unknown)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = calculate_raw_features(inputs, workdir=workdir, radius=radius,
                                 device=device, batch_size=batch_size,
                                 save_intermediates=save_intermediates)

    print("[5/5] Transforming embeddings through the fine-tuned MLP...")
    mpnn_learned = transform_embeddings(raw["mpnn_embeddings"], config.MLP_CHECKPOINT,
                                        config.MLP_CONFIG, device)
    if save_intermediates:
        np.savez_compressed(os.path.join(workdir, "embeddings_pmpnn_learned.npz"),
                            embeddings=mpnn_learned, ids=raw["ids"], labels=raw["labels"])

    print(f"\nFeature matrices ready for {len(raw['ids'])} complex(es):")
    print(f"  MPNN-learned (structural): {mpnn_learned.shape}")
    print(f"  sequence + score:          {raw['seq_combined'].shape}")

    return {
        "ids":          raw["ids"],
        "mpnn_learned": mpnn_learned,
        "seq_combined": raw["seq_combined"],
        "labels":       raw["labels"],
    }
