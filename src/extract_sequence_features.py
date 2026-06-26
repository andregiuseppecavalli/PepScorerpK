"""
Extract per-complex hand-crafted sequence features (peptide + pocket) from PDB
files. Importable form of the original extract_sequence_features.py; residue
roles are now assigned by (chain_id, res_id) keys.

Features per chain (27 per chain, 54 total):
    20  AA composition fractions (A C D E F G H I K L M N P Q R S T V W Y)
    1   length
    1   mean hydrophobicity (Kyte-Doolittle)
    1   mean formal charge at pH ~7
    1   fraction aromatic (F, W, Y, H)
    1   total hydrophobicity
    1   total charge
    1   aromatic count
"""
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdb
from biotite.sequence import ProteinSequence
from tqdm import tqdm


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

KYTE_DOOLITTLE = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}
CHARGE_AT_PH7 = {
    "A":  0.0, "C":  0.0, "D": -1.0, "E": -1.0, "F":  0.0,
    "G":  0.0, "H":  0.1, "I":  0.0, "K": +1.0, "L":  0.0,
    "M":  0.0, "N":  0.0, "P":  0.0, "Q":  0.0, "R": +1.0,
    "S":  0.0, "T":  0.0, "V":  0.0, "W":  0.0, "Y":  0.0,
}
AROMATIC_SET = {"F", "W", "Y", "H"}

FEATURES_PER_CHAIN = 27
TOTAL_FEATURES = 2 * FEATURES_PER_CHAIN  # 54


def build_feature_names():
    names = []
    for chain in ("pep", "pock"):
        names += [f"{chain}_frac_{aa}" for aa in AA_LIST]
        names += [
            f"{chain}_length",
            f"{chain}_mean_hydrophobicity",
            f"{chain}_mean_charge",
            f"{chain}_frac_aromatic",
            f"{chain}_total_hydrophobicity",
            f"{chain}_total_charge",
            f"{chain}_aromatic_count",
        ]
    return names


FEATURE_NAMES = build_feature_names()
assert len(FEATURE_NAMES) == TOTAL_FEATURES


def compute_features(aa_list):
    """Compute 27 features from a list of 1-letter amino acid codes."""
    feats = np.zeros(FEATURES_PER_CHAIN, dtype=np.float32)
    if len(aa_list) == 0:
        return feats
    for aa in aa_list:
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            feats[idx] += 1.0
    feats[:20] /= len(aa_list)
    hydros   = np.array([KYTE_DOOLITTLE.get(aa, 0.0) for aa in aa_list], dtype=np.float32)
    charges  = np.array([CHARGE_AT_PH7.get(aa, 0.0)  for aa in aa_list], dtype=np.float32)
    aromatic = np.array([1.0 if aa in AROMATIC_SET else 0.0 for aa in aa_list], dtype=np.float32)
    feats[20] = float(len(aa_list))
    feats[21] = hydros.mean()
    feats[22] = charges.mean()
    feats[23] = aromatic.mean()
    feats[24] = hydros.sum()
    feats[25] = charges.sum()
    feats[26] = aromatic.sum()
    return feats


def parse_pdb_for_aa_lists(fpath, pep_keys, pock_keys):
    """
    Parse a PDB and return (peptide_aa_list, pocket_aa_list, status_msg).
    Residue roles are assigned by (chain_id, res_id) membership.
    """
    try:
        pdb_file = pdb.PDBFile.read(fpath)
        structure = pdb.get_structure(pdb_file, model=1)
        structure = structure[struc.filter_amino_acids(structure)]
    except Exception as e:
        return None, None, f"Read Error: {e}"

    pep_keys = set(pep_keys) if pep_keys is not None else set()
    pock_keys = set(pock_keys) if pock_keys is not None else set()
    if not pep_keys:
        return None, None, "Empty peptide key set."

    pep_aas, pock_aas = [], []
    residue_starts = struc.get_residue_starts(structure)

    for i in range(len(residue_starts)):
        start = residue_starts[i]
        end = residue_starts[i + 1] if i + 1 < len(residue_starts) else len(structure)
        res = structure[start:end]

        c_id = res.chain_id[0]
        try:
            r_id = int(res.res_id[0])
        except Exception:
            continue
        key = (c_id, r_id)

        try:
            aa_1 = ProteinSequence.convert_letter_3to1(res.res_name[0])
        except Exception:
            aa_1 = "X"

        is_peptide = key in pep_keys
        is_pocket  = (key in pock_keys) and (not is_peptide)
        if is_peptide:
            pep_aas.append(aa_1)
        elif is_pocket:
            pock_aas.append(aa_1)

    if len(pep_aas) == 0:
        return None, None, "No peptide residues matched after parsing."
    if len(pock_aas) == 0:
        return None, None, "No pocket residues matched after parsing."
    return pep_aas, pock_aas, "OK"


def compute_sequence_features(records):
    """
    Args:
        records: list of dicts with keys id, pdb_path, pep_keys, pock_keys,
                 and optional label.

    Returns:
        ids    [N]      complex identifiers (kept rows only)
        X      [N, 54]  sequence features
        labels [N]      pK labels (NaN where unknown)
    """
    all_feats, all_labels, all_ids = [], [], []
    for rec in tqdm(records, desc="Sequence features"):
        pep_aas, pock_aas, status = parse_pdb_for_aa_lists(
            rec["pdb_path"], rec["pep_keys"], rec["pock_keys"]
        )
        if pep_aas is None:
            print(f"  skip {rec['id']}: {status}")
            continue
        feats = np.concatenate(
            [compute_features(pep_aas), compute_features(pock_aas)], axis=0
        )
        all_feats.append(feats)
        all_labels.append(float(rec.get("label", float("nan"))))
        all_ids.append(rec["id"])

    if not all_ids:
        raise RuntimeError("No complexes produced sequence features.")

    X = np.stack(all_feats, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.float32)
    ids = np.array(all_ids)
    return ids, X, y
