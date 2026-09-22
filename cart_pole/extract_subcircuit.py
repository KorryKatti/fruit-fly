"""Option E: extract a sensory-to-motor subcircuit (~50K neurons) from the full connectome.

Seeds: sensory 0:1000 + existing motor pools.
Expand forward (pre->post) from sensory and backward (post->pre) from motor,
union until target size, save induced subgraph.
"""
import numpy as np
import scipy.sparse as sp

FULL_A_PATH = 'connectome_adjacency.npz'
OUT_A = 'subcircuit_A.npz'
OUT_MAP = 'subcircuit_idx.npy'  # kept old indices, sorted; new_i = position
TARGET = 50_000

print("loading full A...", flush=True)
A = sp.load_npz(FULL_A_PATH).tocsr()
n = A.shape[0]
print(f"A {A.shape} nnz={A.nnz}", flush=True)

# A[pre, post]: forward neighbors of i = A[i].indices (posts)
# reverse neighbors of i = A.T[i].indices = column i (pres)
At = A.T.tocsr()

sensory = np.arange(1000, dtype=np.int64)
motor = np.concatenate([
    np.load('data/motor_pos_idx.npy').astype(np.int64),
    np.load('data/motor_neg_idx.npy').astype(np.int64),
])

def expand(frontier, mat, max_add):
    """One hop along mat rows: all mat[f].indices not yet kept."""
    if len(frontier) == 0:
        return np.array([], dtype=np.int64)
    # slice rows frontier -> cols
    neigh = mat[frontier]
    # convert to coo for unique columns efficiently
    cols = neigh.indices  # for CSR slice, indices are column ids of nonzero in those rows
    # note: CSR row slice returns sparse matrix with same cols; .indices gives all col ids
    return cols

kept = np.zeros(n, dtype=bool)
kept[sensory] = True
kept[motor] = True

# frontier for forward expansion from sensory, backward from motor
f_fwd = sensory.copy()
f_bwd = motor.copy()

rng = np.random.default_rng(0)

def add_new(cols):
    new = cols[~kept[cols]]
    # unique
    if new.size == 0:
        return np.array([], dtype=np.int64)
    new = np.unique(new)
    room = TARGET - kept.sum()
    if new.size > room:
        new = rng.choice(new, room, replace=False)
    kept[new] = True
    return new

total_kept = int(kept.sum())
print(f"seeds: {total_kept}", flush=True)

hop = 0
while total_kept < TARGET and hop < 12:
    hop += 1
    # forward hop
    if f_fwd.size:
        # only expand rows that are kept and were frontier
        cols = A[f_fwd].indices
        f_fwd = add_new(cols)
    # backward hop
    if f_bwd.size and total_kept < TARGET:
        cols = At[f_bwd].indices
        f_bwd = add_new(cols)
    total_kept = int(kept.sum())
    print(f"hop {hop}: kept={total_kept}", flush=True)
    if f_fwd.size == 0 and f_bwd.size == 0:
        break

# If still under target (disconnected), fill with random neighbors of kept
if total_kept < TARGET:
    print("filling with neighbor sampling...", flush=True)
    kept_idx = np.flatnonzero(kept)
    while total_kept < TARGET:
        batch = rng.choice(kept_idx, size=min(5000, kept_idx.size), replace=False)
        cols = A[batch].indices
        new = add_new(cols)
        if new.size == 0:
            # last resort: random unused neurons
            room = TARGET - total_kept
            cand = np.flatnonzero(~kept)
            if cand.size == 0:
                break
            take = rng.choice(cand, min(room, cand.size), replace=False)
            kept[take] = True
            new = take
        total_kept = int(kept.sum())
        print(f"  fill kept={total_kept}", flush=True)

kept_idx = np.flatnonzero(kept).astype(np.int64)
print(f"final kept: {kept_idx.size}", flush=True)

print("inducing subgraph...", flush=True)
sub_A = A[kept_idx][:, kept_idx].tocsr()
print(f"sub_A {sub_A.shape} nnz={sub_A.nnz}", flush=True)

np.savez_compressed(OUT_A, data=sub_A.data, indices=sub_A.indices,
                    indptr=sub_A.indptr, shape=np.array(sub_A.shape))
np.save(OUT_MAP, kept_idx)

# remap motor pools; check all present
old_to_new = np.full(n, -1, dtype=np.int64)
old_to_new[kept_idx] = np.arange(kept_idx.size)
pos = old_to_new[np.load('data/motor_pos_idx.npy')]
neg = old_to_new[np.load('data/motor_neg_idx.npy')]
print(f"motor pos in sub: {(pos>=0).sum()}/{pos.size}, neg: {(neg>=0).sum()}/{neg.size}", flush=True)
np.save('subcircuit_motor_pos.npy', pos[pos >= 0])
np.save('subcircuit_motor_neg.npy', neg[neg >= 0])

# sensory 0:1000 all in?
sens_new = old_to_new[sensory]
print(f"sensory in sub: {(sens_new>=0).sum()}/1000", flush=True)
np.save('subcircuit_sensory.npy', sens_new[sens_new >= 0])

print(f"saved {OUT_A}, {OUT_MAP}", flush=True)
print("DONE", flush=True)
