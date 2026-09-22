"""Regenerate motor pools under corrected dynamics (A.T, THR=10, SC=50, 1000 sensory)."""
import numpy as np
import scipy.sparse as sp
from sensors import encode_cartpole_state
from neuron import LIFNeuron

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]
N_SENSORY = 1000

def run(angle, steps=8, decay=0.95, thr=10.0, sens_scale=50.0):
    neuron = LIFNeuron(n, decay=decay, threshold=thr)
    sensory = encode_cartpole_state(np.array([0, 0, angle, 0.0]), n) * sens_scale
    for _ in range(steps):
        spikes = neuron.step(A, sensory)
    return spikes.copy(), neuron.v.copy()

print("finding pools (thr=10, sc=50, steps=8)...")
sp_p, v_p = run(+0.3)
sp_n, v_n = run(-0.3)
sp_z, _ = run(0.0)

diff_s = sp_p - sp_n
diff_s[:N_SENSORY] = 0
print(f"spikes @+0.3: {sp_p.sum():.0f}, @-0.3: {sp_n.sum():.0f}, @0: {sp_z.sum():.0f}")
print(f"spike diff nonzero: {np.count_nonzero(diff_s)}")

pos_idx = np.where(diff_s > 0)[0]
neg_idx = np.where(diff_s < 0)[0]
pos_idx = pos_idx[pos_idx >= N_SENSORY]
neg_idx = neg_idx[neg_idx >= N_SENSORY]
print(f"prefer+: {len(pos_idx)}, prefer-: {len(neg_idx)}")

if len(pos_idx) == 0 or len(neg_idx) == 0:
    print("no pools, abort")
    raise SystemExit

rng = np.random.default_rng(0)
def cap(idx, k=500):
    return rng.choice(idx, k, replace=False) if len(idx) > k else idx

pp, nn = cap(pos_idx), cap(neg_idx)

print("\nfresh angle sweep:")
for ang in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
    s, _ = run(ang)
    ps, ns = s[pp].sum(), s[nn].sum()
    print(f"  {ang:+.1f}: +={ps:4.0f} -={ns:4.0f} act={1 if ps > ns else 0}")

np.save('data/motor_pos_idx.npy', pp)
np.save('data/motor_neg_idx.npy', nn)
print("\nsaved data/motor_pos_idx.npy, data/motor_neg_idx.npy")
