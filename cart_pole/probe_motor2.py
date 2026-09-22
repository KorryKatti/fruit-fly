import numpy as np
import scipy.sparse as sp

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]

# In-degree (incoming weight) per neuron: col sums of A = A.sum(axis=0)
# For CSR, sum(axis=0) is efficient enough
print("Computing in-degree (column sums)...")
in_deg = np.asarray(A.sum(axis=0)).ravel()
out_deg = np.asarray(A.sum(axis=1)).ravel()

print(f"in-degree:  min={in_deg.min():.4g} max={in_deg.max():.4g} mean={in_deg.mean():.4g}")
print(f"out-degree: min={out_deg.min():.4g} max={out_deg.max():.4g} mean={out_deg.mean():.4g}")
print(f"neurons with zero in-degree: {np.count_nonzero(in_deg == 0)} / {n}")
print(f"neurons with zero out-degree: {np.count_nonzero(out_deg == 0)} / {n}")

# Current motor pool in-degree
M0, M1 = 88384000, 88384522
print(f"\ncurrent motor pool in-degree: min={in_deg[M0:M1].min():.4g} max={in_deg[M0:M1].max():.4g} mean={in_deg[M0:M1].mean():.4g}")

# Current sensory pool
S0, S1 = 0, 500
print(f"sensory pool out-degree: min={out_deg[S0:S1].min():.4g} max={out_deg[S0:S1].max():.4g} mean={out_deg[S0:S1].mean():.4g}")

# Find best-connected candidate pools by in-degree
# Look at top in-degree neurons location
top_in = np.argsort(-in_deg)[:1000]
print(f"\ntop-1000 in-degree neuron index range: {top_in.min()} .. {top_in.max()}")
buckets = top_in // 1_000_000
ub, cb = np.unique(buckets, return_counts=True)
order = np.argsort(-cb)[:10]
print("top-1000 in-degree 1M-buckets:")
for i in order:
    print(f"  {ub[i]}M: {cb[i]}")

# Strongest sensory projection targets
print("\nComputing sensory out-projection weight by target...")
sens_out = A[S0:S1, :].tocsr()  # rows = sensory local, cols = global post
# weight into each post from sensory
post_w = np.asarray(sens_out.sum(axis=0)).ravel()
top_post = np.argsort(-post_w)[:2000]
print(f"top sensory-projection targets index range: {top_post.min()} .. {top_post.max()}")
print(f"top target weights: {post_w[top_post[:10]]}")
buckets = top_post // 1_000_000
ub, cb = np.unique(buckets, return_counts=True)
order = np.argsort(-cb)[:10]
print("top sensory-target 1M-buckets:")
for i in order:
    print(f"  {ub[i]}M: {cb[i]}")

# Where do the last 522 neurons rank in sensory drive?
motor_sens_drive = post_w[M0:M1]
print(f"\ncurrent motor pool sensory drive: min={motor_sens_drive.min():.4g} max={motor_sens_drive.max():.4g} sum={motor_sens_drive.sum():.4g}")
print(f"neurons with sensory drive > 0 overall: {np.count_nonzero(post_w)}")

# Distribution of post_w for nonzero
nz = post_w[post_w > 0]
print(f"sensory drive percentiles (nonzero): p50={np.percentile(nz,50):.4g} p90={np.percentile(nz,90):.4g} p99={np.percentile(nz,99):.4g} p99.9={np.percentile(nz,99.9):.4g}")

# Candidate motor pool: high in-degree AND receives sensory drive
score = in_deg * np.log1p(post_w)
# exclude current sensory pool itself
score[S0:S1] = 0
cand = np.argsort(-score)[:522]
print(f"\nbest candidate motor pool by in_deg*sensory_drive: idx {cand.min()}..{cand.max()}")
print(f"  their in-degree: mean={in_deg[cand].mean():.4g}, sensory drive mean={post_w[cand].mean():.4g}")

np.save('/tmp/opencode/post_w.npy', post_w)
np.save('/tmp/opencode/in_deg.npy', in_deg)
print("\nsaved post_w, in_deg")
