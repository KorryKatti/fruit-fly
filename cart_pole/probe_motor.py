import numpy as np
import scipy.sparse as sp

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]
print(f"shape={A.shape}, nnz={A.nnz}, dtype={A.dtype}")
print(f"weight range: min={A.data.min():.4g}, max={A.data.max():.4g}, mean={A.data.mean():.4g}")

# Convention from neuron.py: synaptic = A @ spikes → result[i] = sum_j A[i,j]*spikes[j]
# So A[i,j] = weight from pre j → post i. Columns = pre, rows = post.

M0, M1 = 88384000, 88384522
S0, S1 = 0, 500

# --- motor neurons as POST (columns) ---
motor_in = A[:, M0:M1].tocoo()  # entries where col in motor range
motor_rows_with_in = np.unique(motor_in.row)
motor_cols_with_in = np.unique(motor_in.col)
print(f"\n[motor as POST] incoming nnz: {motor_in.nnz}")
print(f"[motor as POST] motor cells receiving input: {len(motor_cols_with_in)}/{M1-M0}")
print(f"[motor as POST] incoming weight sum: {motor_in.data.sum():.4g}")
if motor_in.nnz:
    pre = motor_in.row
    uniq = np.unique(pre)
    print(f"[motor as POST] unique pre-synaptic: {len(uniq)}, range {uniq.min()}..{uniq.max()}")
    sens_pre = uniq[(uniq >= S0) & (uniq < S1)]
    print(f"[motor as POST] pre in sensory 0:500: {len(sens_pre)}")
    buckets = pre // 1_000_000
    ub, cb = np.unique(buckets, return_counts=True)
    order = np.argsort(-cb)[:10]
    print("[motor as POST] top pre-syn 1M-buckets:")
    for i in order:
        print(f"  {ub[i]}M: {cb[i]}")
else:
    print("[motor as POST] NO incoming synapses")

# --- motor neurons as PRE (rows) ---
motor_out = A[M0:M1, :]
print(f"\n[motor as PRE] outgoing nnz: {motor_out.nnz}, sum: {motor_out.sum():.4g}")

# --- sensory as PRE (rows) ---
sens_out = A[S0:S1, :].tocoo()
print(f"\n[sensory as PRE] outgoing nnz: {sens_out.nnz}, sum: {sens_out.data.sum():.4g}")
if sens_out.nnz:
    post = sens_out.col
    uniq = np.unique(post)
    print(f"[sensory as PRE] unique post-synaptic: {len(uniq)}, range {uniq.min()}..{uniq.max()}")
    to_motor = uniq[(uniq >= M0) & (uniq < M1)]
    print(f"[sensory as PRE] direct projections to motor: {len(to_motor)}")
    buckets = post // 1_000_000
    ub, cb = np.unique(buckets, return_counts=True)
    order = np.argsort(-cb)[:10]
    print("[sensory as PRE] top post-syn 1M-buckets:")
    for i in order:
        print(f"  {ub[i]}M: {cb[i]}")

# --- sensory as POST (columns) ---
sens_in = A[:, S0:S1]
print(f"\n[sensory as POST] incoming nnz: {sens_in.nnz}, sum: {sens_in.sum():.4g}")

# overall sparsity of motor rows/cols
print(f"\n[diag] A[M0:M1,M0:M1] nnz: {A[M0:M1, M0:M1].nnz}")
print(f"[weight percentiles] p50={np.percentile(A.data,50):.4g} p90={np.percentile(A.data,90):.4g} p99={np.percentile(A.data,99):.4g}")
