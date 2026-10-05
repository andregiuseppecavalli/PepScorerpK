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

class ChainExcluder(Select):
    """
    A custom selection class that excludes a specific chain.
    """
    def __init__(self, chain_to_exclude):
        self.chain_to_exclude = chain_to_exclude

    def accept_chain(self, chain):
        # If the chain ID matches the one we want to exclude, return 0 (False)
        if chain.get_id() == self.chain_to_exclude:
            return 0
        # Otherwise, keep it
        return 1

def save_pdb_without_chain(input_path, output_path, chain_to_remove):
    """
    Reads a PDB, removes the specified chain, and saves the result.
    """
    # 1. Initialize Parser
    parser = PDBParser(QUIET=True) # QUIET=True suppresses warnings about PDB headers
    
    try:
        # 2. Read the structure
        structure = parser.get_structure("structure", input_path)
        
        # 3. Check if chain actually exists (optional, for user feedback)
        found = False
        for model in structure:
            if chain_to_remove in [c.id for c in model]:
                found = True
                break
        if not found:
            print(f"Warning: Chain '{chain_to_remove}' was not found in {input_path}, but the file will be saved anyway.")

        # 4. Initialize IO writer
        io = PDBIO()
        io.set_structure(structure)
        
        # 5. Save using the custom Select class
        # This writes the file while filtering out the unwanted chain on the fly
        io.save(output_path, select=ChainExcluder(chain_to_remove))
        
        print(f"Success: Saved '{output_path}' excluding chain '{chain_to_remove}'.")

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def parse_results_symmetric(pdbs, output_dir):
    """
    Parses the upper triangle files and builds a symmetric matrix.
    Diagonal is set to 1.0 automatically.
    
    UPDATED: Now selects the highest TM-score found in the file, 
    regardless of which chain was used for normalization.
    """
    n = len(pdbs)
    similarity_matrix = np.zeros((n, n))
    
    # Regex patterns for TM-scores
    # (We no longer strictly need the length regexes for the decision logic)
    re_tm1 = re.compile(r"TM-score=\s*([\d\.]+)\s*\(if normalized by length of Chain_1")
    re_tm2 = re.compile(r"TM-score=\s*([\d\.]+)\s*\(if normalized by length of Chain_2")

    print("Parsing results and symmetrizing matrix (using Max TM-score)...")

    for i, pdb1 in enumerate(pdbs):
        similarity_matrix[i, i] = 1.0
        for j in range(i+1, n):
            pdb2 = pdbs[j]
            out_file = os.path.join(output_dir, f"{pdb1}_{pdb2}.txt")
            
            if not os.path.exists(out_file):
                print(f"Missing: {out_file}")
                continue
            
            try:
                with open(out_file, 'r') as f:
                    content = f.read()

                # Search for scores
                match_tm1 = re_tm1.search(content)
                match_tm2 = re_tm2.search(content)
                
                # We only proceed if we successfully found BOTH scores
                if match_tm1 and match_tm2:
                    tm1 = float(match_tm1.group(1))
                    tm2 = float(match_tm2.group(1))
                    
                    # --- CHANGED LOGIC ---
                    # Simply take the maximum of the two scores
                    final_score = max(tm1, tm2)
                        
                    # FILL BOTH SIDES OF THE MATRIX
                    similarity_matrix[i, j] = final_score
                    similarity_matrix[j, i] = final_score
                else:
                    print(f"Warning: Could not extract TM-scores from {out_file}")

            except Exception as e:
                print(f"Error parsing {out_file}: {e}")

    return similarity_matrix

def read_mmalign_matrix(matrix_file_path):
    """
    Parses the MM-align output matrix.
    Returns U (3x3 rotation) and t (1x3 translation).
    """
    t_vec = np.zeros(3)
    u_mat = np.zeros((3, 3))
    
    try:
        with open(matrix_file_path, 'r') as f:
            lines = f.readlines()
            
        start_reading = False
        for line in lines:
            if "The rotation matrix to rotate Chain_1 to Chain_2" in line:
                start_reading = True
                continue
            
            if start_reading:
                parts = line.strip().split()
                # Parse rows starting with index 0, 1, 2
                if len(parts) >= 5 and parts[0] in ['0', '1', '2']:
                    row_idx = int(parts[0])
                    t_vec[row_idx] = float(parts[1])
                    u_mat[row_idx, 0] = float(parts[2])
                    u_mat[row_idx, 1] = float(parts[3])
                    u_mat[row_idx, 2] = float(parts[4])
        return u_mat, t_vec
        
    except FileNotFoundError:
        print(f"Matrix file not found: {matrix_file_path}")
        return None, None

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