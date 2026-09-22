import numpy as np
import scipy.sparse as sp
from sensors import encode_cartpole_state
from neuron import LIFNeuron

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]

def run(angle, steps=8, decay=0.95, thr=5.0, sens_scale=10.0):
    neuron = LIFNeuron(n, decay=decay, threshold=thr)
    sensory = encode_cartpole_state(np.array([0,0,angle,0.0]), n) * sens_scale
    for _ in range(steps):
        spikes = neuron.step(A, sensory)
    return spikes.copy(), neuron.v.copy()

print("sweep thr x sens_scale (steps=8):")
for thr, sc in [(2.0,1), (2.0,10), (5.0,10), (5.0,20), (10.0,20), (10.0,50), (20.0,50)]:
    sp_p, _ = run(+0.3, thr=thr, sens_scale=sc)
    sp_n, _ = run(-0.3, thr=thr, sens_scale=sc)
    d = sp_p - sp_n
    d[0:500] = 0
    print(f"  thr={thr:4.1f} sc={sc:3d}: p/n={sp_p.sum():8.0f}/{sp_n.sum():8.0f}, "
          f"diff_nz={np.count_nonzero(d)}, max|d|={np.abs(d).max():.0f}")

# pick promising config
THR, SC = 10.0, 50.0
print(f"\n=== thr={THR}, sens_scale={SC} ===")
sp_p, _ = run(+0.3, thr=THR, sens_scale=SC)
sp_n, _ = run(-0.3, thr=THR, sens_scale=SC)
d = sp_p - sp_n
d[0:500] = 0
pos_idx = np.where(d > 0)[0]
neg_idx = np.where(d < 0)[0]
print(f"prefer+: {len(pos_idx)}, prefer-: {len(neg_idx)}")

if len(pos_idx) == 0 or len(neg_idx) == 0:
    print("no pools, abort")
    raise SystemExit

rng = np.random.default_rng(0)
def cap(idx, k=500):
    return rng.choice(idx, k, replace=False) if len(idx) > k else idx
pp, nn = cap(pos_idx), cap(neg_idx)

print("\nfresh:")
for ang in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
    s, _ = run(ang, thr=THR, sens_scale=SC)
    ps, ns = s[pp].sum(), s[nn].sum()
    print(f"  {ang:+.1f}: +={ps:4.0f} -={ns:4.0f} act={1 if ps>ns else 0}")

print("\ncontinuous:")
neuron = LIFNeuron(n, decay=0.95, threshold=THR)
for i, ang in enumerate([0.0,0.1,0.2,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.2,-0.1,0.0]):
    sensory = encode_cartpole_state(np.array([0,0,ang,0.0]), n) * SC
    spikes = neuron.step(A, sensory)
    ps, ns = spikes[pp].sum(), spikes[nn].sum()
    act = 1 if ps > ns else 0
    print(f"  {i:2d} {ang:+.1f}: +={ps:4.0f} -={ns:4.0f} d={ps-ns:5.0f} act={act}")

np.save('/tmp/opencode/pos4.npy', pp)
np.save('/tmp/opencode/neg4.npy', nn)
np.save('/tmp/opencode/cfg.npy', np.array([THR, SC]))
