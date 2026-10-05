import pandas as pd
import os
import networkx as nx
from tqdm import tqdm

### CONFIGURATION BLOCK ###
DATASET_PATH = "../data/Dataset.csv"
TM_MATRIX_PATH = "../data/TM_matrix.csv"
LIGAND_RMSD_MATRIX_PATH = "../data/Ligand_RMSD_matrix.csv"
LIGAND_SIM_MATRIX_PATH = "../data/Ligand_similarity_matrix.csv"
CLUSTER_COLUMN_NAME = "cluster" # change based on personal preference
OUTPUT_FILE = "../data/Dataset_clustered.csv" # change based on personal preference

# Thresholds
PK_THRESHOLD = 1.0
SEQ_ID_THRESHOLD = 0.9
TM_THRESHOLD = 0.8
COMBINED_THRESHOLD = 0.8

# Load CSV files
df = pd.read_csv(DATASET_PATH)
pdbs = list(df['PDB'].unique())
matrix = pd.read_csv(TM_MATRIX_PATH, index_col=0)
ligad_rmsd_matrix = pd.read_csv(LIGAND_RMSD_MATRIX_PATH, index_col=0)
ligand_sim_matrix = pd.read_csv(LIGAND_SIM_MATRIX_PATH, index_col=0)

# --- 1. Pre-computation Phase ---
# Create dictionary for access inside the loop
pdb_to_pk = df.set_index('PDB')['pK'].to_dict()

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
            
        if abs(pk_i - pk_j) > PK_THRESHOLD:
            print('different pK, no edge')
            continue # Different cluster

        # --- Check 2: Ligand Sequence Similarity (lookup) ---
        pep_identity = ligand_sim_matrix.at[i, j]
        
        if pep_identity > SEQ_ID_THRESHOLD:
            print('peptide similar, added edge')
            G.add_edge(i, j) # Same cluster
            continue # Check satisfied, move to next pair

        # --- Check 3: Protein TM Score ---
        tm_score = matrix.at[i, j]
        
        if tm_score < TM_THRESHOLD:
            print('different proteins, no egde')
            continue # Different cluster

        # --- Check 4: Ligand RMSD Check ---
        
        rmsd = ligad_rmsd_matrix.at[i, j]
        
        # Formula: Similarity + (1 - RMSD) > 0.8
        combined_score = pep_identity + (1 - rmsd)
        
        if combined_score > COMBINED_THRESHOLD:
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