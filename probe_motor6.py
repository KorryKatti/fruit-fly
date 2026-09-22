import numpy as np
import scipy.sparse as sp
from sensors import encode_cartpole_state
from neuron import LIFNeuron

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]

def run(angle, steps=5, decay=0.95, thr=2.0):
    neuron = LIFNeuron(n, decay=decay, threshold=thr)
    state = np.array([0.0, 0.0, angle, 0.0])
    sensory = encode_cartpole_state(state, n)
    for _ in range(steps):
        spikes = neuron.step(A, sensory)
    return spikes.copy(), neuron.v.copy()

# Find differential pools under NEW dynamics
sp_p, v_p = run(+0.3)
sp_n, v_n = run(-0.3)
sp_z, v_z = run(0.0)

# Prefer voltage diff (graded) or spike diff?
diff_v = v_p - v_n
diff_v[0:500] = 0
diff_s = sp_p - sp_n
diff_s[0:500] = 0

print(f"voltage diff nonzero: {np.count_nonzero(np.abs(diff_v)>0.01)}, max={np.abs(diff_v).max():.4g}")
print(f"spike diff nonzero: {np.count_nonzero(diff_s != 0)}")
print(f"spikes @+0.3: {sp_p.sum():.0f}, @-0.3: {sp_n.sum():.0f}, @0: {sp_z.sum():.0f}")

# Build pools from spike preference (what decoder sees)
pos_idx = np.where((sp_p - sp_n) > 0)[0]
neg_idx = np.where((sp_p - sp_n) < 0)[0]
pos_idx = pos_idx[pos_idx >= 500]
neg_idx = neg_idx[neg_idx >= 500]
print(f"\nspike-prefer+ : {len(pos_idx)}, prefer-: {len(neg_idx)}")

# Also try voltage-preference pools
vpos = np.where((v_p - v_n) > 0.5)[0]
vneg = np.where((v_p - v_n) < -0.5)[0]
vpos = vpos[vpos >= 500]
vneg = vneg[vneg >= 500]
print(f"volt-prefer+ : {len(vpos)}, prefer-: {len(vneg)}")

rng = np.random.default_rng(0)
def cap(idx, k=500):
    if len(idx) > k: return rng.choice(idx, k, replace=False)
    return idx

for name, pp, nn in [("spike", cap(pos_idx), cap(neg_idx)),
                     ("volt", cap(vpos), cap(vneg))]:
    if len(pp) == 0 or len(nn) == 0:
        continue
    print(f"\n=== {name} pools ===")
    print("angle  | sp+  sp-  d  a | v+     v-     dv    a")
    for ang in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
        s, vv = run(ang)
        ps, ns = s[pp].sum(), s[nn].sum()
        pv, nv = vv[pp].mean(), vv[nn].mean()
        a1 = 1 if ps > ns else 0
        a2 = 1 if pv > nv else 0
        print(f"{ang:+.1f} | {ps:4.0f} {ns:4.0f} {ps-ns:4.0f} {a1} | {pv:6.1f} {nv:6.1f} {pv-nv:6.1f} {a2}")

# Continuous run test with best-looking pool (voltage)
print("\n=== continuous with volt pools ===")
pp, nn = cap(vpos), cap(vneg)
neuron = LIFNeuron(n, decay=0.95, threshold=2.0)
for i, ang in enumerate([0.0,0.1,0.2,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.2,-0.1,0.0]):
    sensory = encode_cartpole_state(np.array([0,0,ang,0.0]), n)
    spikes = neuron.step(A, sensory)
    pv, nv = neuron.v[pp].mean(), neuron.v[nn].mean()
    ps, ns = spikes[pp].sum(), spikes[nn].sum()
    act_v = 1 if pv > nv else 0
    act_s = 1 if ps > ns else 0
    print(f"{i:2d} {ang:+.1f} | v:{pv:7.1f}-{nv:7.1f}={pv-nv:7.1f} a={act_v} | s:{ps:3.0f}-{ns:3.0f} a={act_s}")

np.save('/tmp/opencode/pos2.npy', pp)
np.save('/tmp/opencode/neg2.npy', nn)
