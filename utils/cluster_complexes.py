import pandas as pd
import numpy as np
import os
import biotite.structure as struc
import biotite.structure.io as struc_io
from biotite.structure.io import pdb as biopdb
from biotite.sequence import ProteinSequence

from Bio import pairwise2
from Bio.Align import substitution_matrices
import networkx as nx
from Bio.PDB import PDBParser, PDBIO, Select
from tqdm import tqdm
import re
from scipy.spatial.distance import cdist

### CONFIGURATION BLOCK ###
DATASET_PATH = "../data/Dataset.csv"
TM_MATRIX_PATH = "../data/TM_matrix.csv"
PDBS_DIR_PATH = "../data/Test_min_pdbs"
LIGAND_RMSD_MATRIX_PATH = "../data/Ligand_RMSD_matrix.csv"
CLUSTER_COLUMN_NAME = "cluster" # change based on personal preference
OUTPUT_FILE = "../data/Dataset_clustered.csv" # change based on personal preference

def compute_identity(seq1, seq2):
    """Compute global sequence identity between two sequences using BLOSUM62."""
    matrix = substitution_matrices.load("BLOSUM62")
   
    alignments = pairwise2.align.globalds(seq1, seq2, matrix, -11, -1, one_alignment_only=True)
    aln1, aln2, score, start, end = alignments[0]
   
    matches = sum(res1 == res2 for res1, res2 in zip(aln1, aln2))
    identity = matches / max(len(seq1), len(seq2))
    return identity

def get_chain_sequence(file_path, chain_id, model_index=1):
    """
    Extracts the amino acid sequence for a specific chain from a PDB file.
    
    Parameters:
        file_path (str): Path to the .pdb file.
        chain_id (str): The specific chain identifier (e.g., 'A', 'B').
        model_index (int): The model number to extract (default is 1).
        
    Returns:
        str: The 1-letter amino acid sequence.
    
    Raises:
        ValueError: If the chain is not found in the structure.
    """
    # 1. Load the PDB file
    pdb_file = biopdb.PDBFile.read(file_path)
    
    # 2. Get the structure (Model 1)
    atom_array = biopdb.get_structure(pdb_file, model=model_index)
    
    # 3. Filter for amino acids only (to remove water/ligands)
    protein_mask = struc.filter_amino_acids(atom_array)
    protein_struct = atom_array[protein_mask]
    
    # 4. Filter for the specific chain ID
    # Note: PDB chain IDs are case-sensitive
    chain_mask = (protein_struct.chain_id == chain_id)
    chain_atoms = protein_struct[chain_mask]
    
    # Check if the chain exists/has protein atoms
    if chain_atoms.array_length() == 0:
        available_chains = sorted(list(set(protein_struct.chain_id)))
        raise ValueError(f"Chain '{chain_id}' not found in protein structure. Available chains: {available_chains}")

    # 5. Extract Residues using residue_iter (Safe Method)
    # This iterates over residues one by one to get the 3-letter code
    three_letter_codes = [res.res_name[0] for res in struc.residue_iter(chain_atoms)]
    
    # 6. Convert to 1-letter sequence
    seq_obj = ProteinSequence(three_letter_codes)
    
    return str(seq_obj)

# Load CSV files
df = pd.read_csv(DATASET_PATH)
pdbs = list(df['PDB'].unique())
matrix = pd.read_csv(TM_MATRIX_PATH)
matrix = matrix.loc[pdbs,pdbs]
ligad_rmsd_matrix = pd.read_csv(LIGAND_RMSD_MATRIX_PATH)

# --- 1. Pre-computation Phase ---
# Create dictionaries for access inside the loop
pdb_to_pk = df.set_index('PDB')['pK'].to_dict()
pdb_to_chain = df.set_index('PDB')['peptide_chain'].to_dict()

# Pre-calculate paths and sequences
pdb_data = {}
for pdb in pdbs:
    path = os.path.join(PDBS_DIR_PATH, f'{pdb}.pdb')
    
    # Get Chain ID and Sequence
    pep_id = pdb_to_chain.get(pdb)
    seq = get_chain_sequence(path, pep_id) 
    
    pdb_data[pdb] = {
        'path': path,
        'pep_id': pep_id,
        'seq': seq
    }

# --- 2. Clustering Loop ---
G = nx.Graph()
G.add_nodes_from(pdbs)
n = len(pdbs)

for a in tqdm(range(n)):
    i = pdbs[a]
    for b in range(a+1, n):
        j = pdbs[b]
        print(f'comparing {i} and {j}')
        # --- Check 1: Delta pK ---
        pk_i = pdb_to_pk.get(i)
        pk_j = pdb_to_pk.get(j)
            
        if abs(pk_i - pk_j) > 1.0:
            print('different pK, no edge')
            continue # Different cluster

        # --- Check 2: Ligand Sequence Similarity ---
        seq_i = pdb_data[i]['seq']
        seq_j = pdb_data[j]['seq']
        
        # Assuming compute_identity returns a float 0.0 to 1.0
        pep_identity = compute_identity(seq_i, seq_j)
        
        if pep_identity > 0.9:
            print('peptide similar, added edge')
            G.add_edge(i, j) # Same cluster
            continue # Check satisfied, move to next pair

        # --- Check 3: Protein TM Score ---
        tm_score = matrix.at[i, j]
        
        if tm_score < 0.8:
            print('different proteins, no egde')
            continue # Different cluster

        # --- Check 4: Ligand RMSD Check ---
        
        rmsd = ligad_rmsd_matrix.at[i, j]
        
        # Formula: Similarity + (1 - RMSD) > 0.8
        combined_score = pep_identity + (1 - rmsd)
        
        if combined_score > 0.8:
            print('similar proteins and similar binding mode, added edge')
            G.add_edge(i, j) # Same cluster

# 4) set clusters and display the result
clusters = list(nx.connected_components(G))

for idx, comp in enumerate(clusters, 1):
    print(f"Cluster {idx} ({len(comp)} members):")
    print(" ", comp)

# 5) set the cluster number in the dataset file
for index in tqdm(df.index):
    pdb = df.at[index, 'PDB']
    cluster_n = None
    for i, cluster in enumerate(clusters):
        if pdb in cluster:
            cluster_n = i+1
            break
    df.at[index, CLUSTER_COLUMN_NAME] = cluster_n

df.to_csv(os.path.join(os.getcwd(), OUTPUT_FILE), index=False)