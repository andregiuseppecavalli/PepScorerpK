"""
Chain-aware identification of peptide and binding-pocket residues.

Replaces the old pep_res_dict / pocket_res_dict pickle dictionaries. The
peptide is now specified by a single chain ID (e.g. "C"); every standard
residue in that chain becomes the peptide, and the pocket is whatever protein
residues fall within `radius` Angstroms of any peptide atom.

IMPORTANT -- residue-ID collisions:
    When a PDB numbers every chain from 1, a peptide residue and a pocket
    residue can share the same integer res_id. To keep them distinct, every
    residue is identified by a (chain_id, res_id) tuple rather than res_id
    alone. The whole downstream pipeline keys its peptide/pocket masks on
    these tuples, so collisions are impossible.

This is a chain-aware adaptation of the get_binding_pocket() helper.
"""
from Bio.PDB import PDBParser, NeighborSearch


def identify_peptide_and_pocket(pdb_file, peptide_chain, radius=5.0):
    """
    Args:
        pdb_file (str):       Path to the PDB file.
        peptide_chain (str):  Chain ID holding the peptide (e.g. "C").
        radius (float):       Distance cutoff in Angstroms (default 5.0).

    Returns:
        (pep_keys, pock_keys): two sets of (chain_id, res_id) tuples.
            pep_keys  -- every standard residue in `peptide_chain`
            pock_keys -- protein residues (any other chain) within `radius`
                         of a peptide atom; waters/heteroatoms excluded.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)
    model = structure[0]  # first model only

    chain_ids = [c.id for c in model]
    if peptide_chain not in chain_ids:
        raise ValueError(
            f"Peptide chain '{peptide_chain}' not found in {pdb_file}. "
            f"Available chains: {chain_ids}"
        )

    # --- Peptide: all standard residues in the requested chain --------------
    peptide_residue_objects = set()
    peptide_atoms = []
    pep_keys = set()

    for residue in model.get_residues():
        chain_id = residue.get_parent().id
        hetflag, resseq, icode = residue.get_id()
        if chain_id == peptide_chain and hetflag == " ":
            peptide_residue_objects.add(residue)
            peptide_atoms.extend(residue.get_atoms())
            pep_keys.add((chain_id, int(resseq)))

    if not peptide_atoms:
        raise ValueError(
            f"No standard residues found in chain '{peptide_chain}'. "
            "Check the chain ID and the PDB file."
        )

    # --- Pocket: protein residues near the peptide, on other chains ---------
    all_atoms = list(model.get_atoms())
    ns = NeighborSearch(all_atoms)

    pock_keys = set()
    for p_atom in peptide_atoms:
        for res in ns.search(p_atom.coord, radius, level="R"):
            if res in peptide_residue_objects:
                continue
            hetflag, resseq, icode = res.get_id()
            if hetflag != " ":              # skip waters / heteroatoms
                continue
            chain_id = res.get_parent().id
            if chain_id == peptide_chain:    # safety: never put peptide in pocket
                continue
            pock_keys.add((chain_id, int(resseq)))

    if not pock_keys:
        raise ValueError(
            f"Empty pocket: no protein residues within {radius} A of chain "
            f"'{peptide_chain}'. Try a larger --radius or check the structure."
        )

    return pep_keys, pock_keys
