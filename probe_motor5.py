import numpy as np
import scipy.sparse as sp
from sensors import encode_cartpole_state

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]

pos_idx = np.load('/tmp/opencode/pos_idx.npy')
neg_idx = np.load('/tmp/opencode/neg_idx.npy')

# continuous run like main.py: no reset, fresh neuron
v = np.zeros(n, dtype=np.float32)
spikes = np.zeros(n, dtype=np.float32)
decay, thr = 0.95, 2.0

# Simulate a sequence of angles crossing zero
angles = [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.2, -0.1, 0.0]

print("Continuous run (no reset):")
print("step  angle    +spk  -spk   diff  act")
for i, ang in enumerate(angles):
    state = np.array([0.0, 0.0, ang, 0.0])
    sensory = encode_cartpole_state(state, n)
    syn = A @ spikes
    v = decay * v + syn + sensory
    spikes = (v > thr).astype(np.float32)

    ps = spikes[pos_idx].sum()
    ns = spikes[neg_idx].sum()
    act = 1 if ps > ns else 0
    print(f"{i:3d}  {ang:+.1f}  {ps:5.0f} {ns:5.0f} {ps-ns:6.0f}  {act}")

# Check total activity (saturation level)
print(f"\nfinal total spikes: {spikes.sum():.0f}")
print(f"final mean voltage: {v.mean():.2f}")

# Also: what if we only count spikes in the current step vs cumulative?
# Already doing current-step spikes. Good.

# Alternative: use voltage mean diff (graded, may survive saturation better)
print("\nSame run, voltage-mean readout:")
v2 = np.zeros(n, dtype=np.float32)
spikes2 = np.zeros(n, dtype=np.float32)
for i, ang in enumerate(angles):
    state = np.array([0.0, 0.0, ang, 0.0])
    sensory = encode_cartpole_state(state, n)
    syn = A @ spikes2
    v2 = decay * v2 + syn + sensory
    spikes2 = (v2 > thr).astype(np.float32)
    pv = v2[pos_idx].mean()
    nv = v2[neg_idx].mean()
    act = 1 if pv > nv else 0
    print(f"{i:3d}  {ang:+.1f}  {pv:7.1f} {nv:7.1f} {pv-nv:7.1f}  {act}")
