"""E2: denser subcircuit via greedy expansion (add nodes with most edges to kept set).

Also reports pathway stats. Overwrites subcircuit_*.npy / subcircuit_A.npz.
"""
import numpy as np
import scipy.sparse as sp

FULL_A_PATH = 'connectome_adjacency.npz'
OUT_A = 'subcircuit_A.npz'
OUT_MAP = 'subcircuit_idx.npy'
TARGET = 50_000

print("loading full A...", flush=True)
A = sp.load_npz(FULL_A_PATH).tocsr()
n = A.shape[0]
At = A.T.tocsr()
print(f"A {A.shape} nnz={A.nnz}", flush=True)

sensory = np.arange(1000, dtype=np.int64)
motor = np.concatenate([
    np.load('data/motor_pos_idx.npy').astype(np.int64),
    np.load('data/motor_neg_idx.npy').astype(np.int64),
])

kept = np.zeros(n, dtype=bool)
kept[sensory] = True
kept[motor] = True
kept_count = int(kept.sum())
print(f"seeds: {kept_count}", flush=True)

rng = np.random.default_rng(0)

# Pathway masks: soft preference (bounded hops so we don't explode)
def bfs_mask(seeds, mat, hops, cap=2_000_000):
    seen = np.zeros(n, dtype=bool)
    seen[seeds] = True
    frontier = seeds.copy()
    for _ in range(hops):
        if frontier.size == 0 or seen.sum() > cap:
            break
        # unique neighbors of frontier rows
        cols = mat[frontier].indices
        new = cols[~seen[cols]]
        if new.size == 0:
            break
        new = np.unique(new)
        # cap frontier growth
        if new.size > 500_000:
            new = rng.choice(new, 500_000, replace=False)
        seen[new] = True
        frontier = new
    return seen

print("forward mask from sensory (4 hops)...", flush=True)
F = bfs_mask(sensory, A, hops=4)
print(f"  F={F.sum()}", flush=True)
print("backward mask to motor (4 hops)...", flush=True)
B = bfs_mask(motor, At, hops=4)
print(f"  B={B.sum()}", flush=True)
on_path = F & B
print(f"on_path F∩B={on_path.sum()}", flush=True)

# Greedy: each round, score non-kept neighbors of kept by (edges_to_kept + path_bonus)
PATH_BONUS = 5.0
rounds = 8
for r in range(rounds):
    room = TARGET - kept_count
    if room <= 0:
        break
    kept_idx = np.flatnonzero(kept)
    # edges from kept -> candidates: slice rows kept_idx (could be large; batch if needed)
    # For up to 50K rows CSR slice is OK
    sub = A[kept_idx]
    # count edges to each destination
    dest = sub.indices
    # filter already kept
    dest = dest[~kept[dest]]
    if dest.size == 0:
        print(f"round {r}: no candidates", flush=True)
        break
    # score via bincount
    max_d = dest.max() + 1
    scores = np.bincount(dest, minlength=max_d).astype(np.float64)
    if scores.size < n:
        scores = np.concatenate([scores, np.zeros(n - scores.size, dtype=np.float64)])
    # path bonus
    scores[on_path] += PATH_BONUS
    # only candidates with score > 0 and not kept
    cand = np.flatnonzero((scores > 0) & (~kept))
    # take top `room` (or all remaining rounds share)
    take = min(room, cand.size)
    # partial sort
    if take < cand.size:
        part = np.argpartition(scores[cand], -take)[-take:]
        chosen = cand[part]
    else:
        chosen = cand
    kept[chosen] = True
    kept_count = int(kept.sum())
    print(f"round {r}: +{chosen.size} -> kept={kept_count}, nnz_so_far~check", flush=True)

# If under target, fill greedily with any remaining high-score or random neighbors
if kept_count < TARGET:
    print("filling...", flush=True)
    kept_idx = np.flatnonzero(kept)
    while kept_count < TARGET:
        sub = A[kept_idx]
        dest = sub.indices
        dest = dest[~kept[dest]]
        if dest.size == 0:
            cand = np.flatnonzero(~kept)
            take = rng.choice(cand, min(TARGET - kept_count, cand.size), replace=False)
            kept[take] = True
        else:
            max_d = dest.max() + 1
            scores = np.bincount(dest, minlength=max_d).astype(np.float64)
            if scores.size < n:
                scores = np.concatenate([scores, np.zeros(n - scores.size)])
            scores[on_path] += PATH_BONUS
            cand = np.flatnonzero((scores > 0) & (~kept))
            room = TARGET - kept_count
            take = min(room, cand.size)
            if take < cand.size:
                part = np.argpartition(scores[cand], -take)[-take:]
                chosen = cand[part]
            else:
                chosen = cand
            kept[chosen] = True
        kept_count = int(kept.sum())
        kept_idx = np.flatnonzero(kept)
        print(f"  fill kept={kept_count}", flush=True)

kept_idx = np.flatnonzero(kept).astype(np.int64)
print(f"final kept: {kept_idx.size}", flush=True)

print("inducing subgraph...", flush=True)
sub_A = A[kept_idx][:, kept_idx].tocsr()
print(f"sub_A {sub_A.shape} nnz={sub_A.nnz} avg_degree={sub_A.nnz/sub_A.shape[0]:.1f}", flush=True)

# pathway coverage inside subgraph
old_to_new = np.full(n, -1, dtype=np.int64)
old_to_new[kept_idx] = np.arange(kept_idx.size)
sens_new = old_to_new[sensory]
pos_old = np.load('data/motor_pos_idx.npy')
neg_old = np.load('data/motor_neg_idx.npy')
pos = old_to_new[pos_old]
neg = old_to_new[neg_old]
print(f"sensory in sub: {(sens_new>=0).sum()}/1000, motor pos: {(pos>=0).sum()}/500 neg: {(neg>=0).sum()}/500", flush=True)
print(f"kept that were on_path: {on_path[kept_idx].sum()}", flush=True)

# reachability: forward from sensory in sub
frontier = set(sens_new[sens_new >= 0].tolist())
seen = np.zeros(sub_A.shape[0], dtype=bool)
seen[sens_new[sens_new >= 0]] = True
motor_new = np.concatenate([pos[pos >= 0], neg[neg >= 0]])
for hop in range(6):
    nxt = set()
    for u in frontier:
        s, e = sub_A.indptr[u], sub_A.indptr[u + 1]
        for v in sub_A.indices[s:e]:
            if not seen[v]:
                seen[v] = True
                nxt.add(int(v))
    frontier = nxt
    hit = int(seen[motor_new].sum())
    print(f"  hop {hop+1}: seen={seen.sum()} motor_reached={hit}/{motor_new.size}", flush=True)
    if not frontier:
        break

np.savez_compressed(OUT_A, data=sub_A.data, indices=sub_A.indices,
                    indptr=sub_A.indptr, shape=np.array(sub_A.shape))
np.save(OUT_MAP, kept_idx)
np.save('subcircuit_motor_pos.npy', pos[pos >= 0])
np.save('subcircuit_motor_neg.npy', neg[neg >= 0])
np.save('subcircuit_sensory.npy', sens_new[sens_new >= 0])
print(f"saved {OUT_A}", flush=True)
print("DONE", flush=True)
