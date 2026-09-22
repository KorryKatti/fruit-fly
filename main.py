import pandas as pd
import scipy.sparse as sp
import numpy as np

# load connections
df = pd.read_feather('connectome-weights-male-cns-v1.0-minconf-0.5.feather')

print(f"Shape : {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Remap IDs to sequential
all_ids = pd.concat([df['body_pre'], df['body_post']]).unique()
id_to_idx = {seg_id: idx for idx, seg_id in enumerate(sorted(all_ids))}

df['pre_idx'] = df['body_pre'].map(id_to_idx)
df['post_idx'] = df['body_post'].map(id_to_idx)

# build sparse adjacency matrix
# A[i,j] = weight from neuron i to neuron j
n_neurons = len(id_to_idx)
A = sp.csr_matrix(
        (df['weight'].values,(df['pre_idx'].values,df['post_idx'].values)),
        shape=(n_neurons,n_neurons),
        dtype=np.float32
        )

print(f"Adjacency matrix shape: {A.shape}")
print(f"Number of synapses: {A.nnz}")
print(f"Memory usage: {A.data.nbytes / 1e9:.2f} GB")

# Save for later use
sp.save_npz('connectome_adjacency.npz', A)
