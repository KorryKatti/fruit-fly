import numpy as np
import scipy.sparse as sp
from sensors import encode_cartpole_state
from neuron import LIFNeuron

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]

def run(angle, steps=5, decay=0.95, thr=2.0):
    neuron = LIFNeuron(n, decay=decay, threshold=thr)
    sensory = encode_cartpole_state(np.array([0,0,angle,0.0]), n)
    for _ in range(steps):
        spikes = neuron.step(A, sensory)
    return spikes.copy(), neuron.v.copy()

# sweep thresholds
for thr in [2.0, 5.0, 10.0, 20.0, 50.0]:
    sp_p, _ = run(+0.3, thr=thr)
    sp_n, _ = run(-0.3, thr=thr)
    sp_z, _ = run(0.0, thr=thr)
    # spike diff
    d = sp_p - sp_n
    d[0:500] = 0
    nz = np.count_nonzero(d)
    print(f"thr={thr:5.1f}: spikes p/n/z = {sp_p.sum():8.0f}/{sp_n.sum():8.0f}/{sp_z.sum():8.0f}, "
          f"diff neurons={nz}, max|d|={np.abs(d).max():.0f}")

# pick a good thr, find pools, test continuous
THR = 20.0
print(f"\n=== thr={THR} pool selection ===")
sp_p, v_p = run(+0.3, thr=THR)
sp_n, v_n = run(-0.3, thr=THR)
d = sp_p - sp_n
d[0:500] = 0
pos_idx = np.where(d > 0)[0]
neg_idx = np.where(d < 0)[0]
print(f"prefer+: {len(pos_idx)}, prefer-: {len(neg_idx)}")

rng = np.random.default_rng(0)
def cap(idx, k=500):
    return rng.choice(idx, k, replace=False) if len(idx) > k else idx

pp, nn = cap(pos_idx), cap(neg_idx)

print("\nfresh runs:")
for ang in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
    s, _ = run(ang, thr=THR)
    ps, ns = s[pp].sum(), s[nn].sum()
    print(f"  {ang:+.1f}: sp+={ps:4.0f} sp-={ns:4.0f} act={1 if ps>ns else 0}")

print("\ncontinuous:")
neuron = LIFNeuron(n, decay=0.95, threshold=THR)
for i, ang in enumerate([0.0,0.1,0.2,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.2,-0.1,0.0]):
    sensory = encode_cartpole_state(np.array([0,0,ang,0.0]), n)
    spikes = neuron.step(A, sensory)
    ps, ns = spikes[pp].sum(), spikes[nn].sum()
    act = 1 if ps > ns else 0
    print(f"  {i:2d} {ang:+.1f}: sp+={ps:4.0f} sp-={ns:4.0f} d={ps-ns:5.0f} act={act}")

np.save('/tmp/opencode/pos3.npy', pp)
np.save('/tmp/opencode/neg3.npy', nn)
np.save('/tmp/opencode/thr.npy', np.array([THR]))
