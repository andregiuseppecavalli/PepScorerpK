import torch
from torch.utils.data import Dataset
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdb
from biotite.sequence import ProteinSequence


# ProteinMPNN amino acid alphabet (21 letters; X = unknown / padding token).
# The order MUST match protein_mpnn_utils.py so that S indices line up with
# the rows of W_s and the columns of W_out logits.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
MPNN_AA_TO_IDX = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}
MPNN_X_IDX = MPNN_AA_TO_IDX["X"]  # 20


def parse_pdb_mpnn(fpath, pep_keys, pock_keys):
    """
    Parse a PDB for ProteinMPNN: backbone N, CA, C, O coordinates, chain
    encoding, residue indices, peptide/pocket masks, and the integer-encoded
    sequence S.

    Args:
        fpath (str):      path to the PDB file
        pep_keys (set):   set of (chain_id, res_id) tuples for peptide residues
        pock_keys (set):  set of (chain_id, res_id) tuples for pocket residues

    Residues are identified by the (chain_id, res_id) tuple so that peptide and
    pocket residues sharing the same integer id (per-chain numbering from 1)
    are never confused.
    """
    try:
        pdb_file = pdb.PDBFile.read(fpath)
        structure = pdb.get_structure(pdb_file, model=1)
        structure = structure[struc.filter_amino_acids(structure)]
    except Exception as e:
        return None, f"Read Error: {e}"

    pep_keys = set(pep_keys) if pep_keys is not None else set()
    pock_keys = set(pock_keys) if pock_keys is not None else set()

    if not pep_keys:
        return None, "Empty peptide key set provided to parser."

    coords         = []
    chain_encoding = []
    res_idx        = []
    mask_pep       = []
    mask_pock      = []
    S_seq          = []

    chain_ids = sorted(set(structure.chain_id))
    chain_to_int = {c: i + 1 for i, c in enumerate(chain_ids)}

    residue_starts = struc.get_residue_starts(structure)

    for i in range(len(residue_starts)):
        start = residue_starts[i]
        end = residue_starts[i + 1] if i + 1 < len(residue_starts) else len(structure)
        res = structure[start:end]

        c_id = res.chain_id[0]
        r_id = int(res.res_id[0])
        key = (c_id, r_id)

        # Backbone atoms ---------------------------------------------------
        try:
            n  = res[res.atom_name == "N"][0].coord
            ca = res[res.atom_name == "CA"][0].coord
            c  = res[res.atom_name == "C"][0].coord
            o_atoms = res[np.isin(res.atom_name, ["O", "OXT"])]
            if len(o_atoms) == 0:
                continue  # missing backbone, drop this residue
            o = o_atoms[0].coord
            coords.append([n, ca, c, o])
        except IndexError:
            continue

        # Sequence ---------------------------------------------------------
        try:
            aa1 = ProteinSequence.convert_letter_3to1(res.res_name[0])
        except Exception:
            aa1 = "X"
        if aa1 not in MPNN_AA_TO_IDX:
            aa1 = "X"
        S_seq.append(MPNN_AA_TO_IDX[aa1])

        # Bookkeeping ------------------------------------------------------
        chain_encoding.append(chain_to_int[c_id])
        res_idx.append(r_id)

        # Role assignment keyed on (chain, res_id)
        is_peptide = key in pep_keys
        is_pocket  = (key in pock_keys) and (not is_peptide)

        if is_peptide:
            mask_pep.append(1.0); mask_pock.append(0.0)
        elif is_pocket:
            mask_pep.append(0.0); mask_pock.append(1.0)
        else:
            mask_pep.append(0.0); mask_pock.append(0.0)

    if len(coords) == 0:
        return None, "Empty structure (no valid N-CA-C-O backbones found)"
    if sum(mask_pep) == 0:
        return None, "Peptide mask empty (peptide keys didn't match the PDB)"
    if sum(mask_pock) == 0:
        return None, "Pocket mask empty (pocket keys didn't match the PDB)"

    return {
        "X":           torch.tensor(np.array(coords), dtype=torch.float32),  # [L, 4, 3]
        "chain_M":     torch.tensor(chain_encoding, dtype=torch.long),
        "residue_idx": torch.tensor(res_idx, dtype=torch.long),
        "mask_pep":    torch.tensor(mask_pep, dtype=torch.float32),
        "mask_pock":   torch.tensor(mask_pock, dtype=torch.float32),
        "S":           torch.tensor(S_seq, dtype=torch.long),
    }, "OK"


class MPNNDataset(Dataset):
    """
    In-memory dataset built from a list of records. Each record is a dict:

        {
            "id":        str,                       # complex identifier
            "pdb_path":  str,                       # path to the PDB
            "pep_keys":  set[(chain_id, res_id)],   # peptide residues
            "pock_keys": set[(chain_id, res_id)],   # pocket residues
            "label":     float (optional, NaN if unknown),
        }
    """

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        data, status = parse_pdb_mpnn(rec["pdb_path"], rec["pep_keys"], rec["pock_keys"])
        if data is None:
            print(f"Skipping {rec['id']}: {status}")
            return None
        label = rec.get("label", float("nan"))
        data["label"] = torch.tensor(float(label), dtype=torch.float32)
        data["id"] = rec["id"]
        return data


def collate_mpnn(batch):
    """Collate variable-length samples into padded tensors."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_len = max(b["X"].shape[0] for b in batch)
    B = len(batch)

    X           = torch.zeros(B, max_len, 4, 3)
    chain_M     = torch.zeros(B, max_len).long()
    residue_idx = torch.zeros(B, max_len).long()
    mask        = torch.zeros(B, max_len)
    m_pep       = torch.zeros(B, max_len)
    m_pock      = torch.zeros(B, max_len)
    S           = torch.full((B, max_len), MPNN_X_IDX, dtype=torch.long)

    labels, ids = [], []
    for i, b in enumerate(batch):
        L = b["X"].shape[0]
        X[i, :L]           = b["X"]
        chain_M[i, :L]     = b["chain_M"]
        residue_idx[i, :L] = b["residue_idx"]
        mask[i, :L]        = 1.0
        m_pep[i, :L]       = b["mask_pep"]
        m_pock[i, :L]      = b["mask_pock"]
        S[i, :L]           = b["S"]
        labels.append(b["label"])
        ids.append(b["id"])

    return X, chain_M, residue_idx, mask, m_pep, m_pock, S, torch.stack(labels), ids
