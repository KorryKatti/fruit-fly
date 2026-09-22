"""Re-find motor pools on the subcircuit under current A.T dynamics."""
import numpy as np
import scipy.sparse as sp
from neuron import LIFNeuron

z = np.load('subcircuit_A.npz')
A = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
n = A.shape[0]
sensory_map = np.load('subcircuit_sensory.npy')
N_SENS = sensory_map.max() + 1  # not contiguous — zero out via map

def encode_sub(angle, ang_vel=0.0, cart_pos=0.0, cart_vel=0.0, scale=1.0):
    from sensors import encode_cartpole_state
    full = encode_cartpole_state(np.array([cart_pos, cart_vel, angle, ang_vel]), 1000)
    s = np.zeros(n, dtype=np.float32)
    s[sensory_map] = full
    return s * scale

def run(angle, steps=8, thr=10.0, sc=50.0):
    neuron = LIFNeuron(n, decay=0.95, threshold=thr)
    sensory = encode_sub(angle, scale=sc)
    for _ in range(steps):
        spikes = neuron.step(A, sensory)
    return spikes.copy()

print("sweep thr x sc on subcircuit:", flush=True)
best = None
for thr, sc in [(2.0, 50), (5.0, 50), (10.0, 50), (10.0, 20), (5.0, 20), (2.0, 20), (10.0, 100)]:
    sp_p = run(+0.3, thr=thr, sc=sc)
    sp_n = run(-0.3, thr=thr, sc=sc)
    d = sp_p - sp_n
    d[sensory_map] = 0
    nz = np.count_nonzero(d)
    print(f"  thr={thr} sc={sc}: p/n={sp_p.sum():6.0f}/{sp_n.sum():6.0f} diff_nz={nz}", flush=True)
    if best is None or nz > best[0]:
        best = (nz, thr, sc)

nz, THR, SC = best
print(f"\nbest diff_nz={nz} at thr={THR} sc={SC}", flush=True)

sp_p = run(+0.3, thr=THR, sc=SC)
sp_n = run(-0.3, thr=THR, sc=SC)
d = sp_p - sp_n
d[sensory_map] = 0
pos_idx = np.where(d > 0)[0]
neg_idx = np.where(d < 0)[0]
# drop sensory
mask_p = ~np.isin(pos_idx, sensory_map)
mask_n = ~np.isin(neg_idx, sensory_map)
pos_idx, neg_idx = pos_idx[mask_p], neg_idx[mask_n]
print(f"prefer+: {len(pos_idx)}, prefer-: {len(neg_idx)}", flush=True)

if len(pos_idx) == 0 or len(neg_idx) == 0:
    print("NO POOLS")
    raise SystemExit(1)

rng = np.random.default_rng(0)
def cap(idx, k=500):
    return rng.choice(idx, k, replace=False) if len(idx) > k else idx
pp, nn = cap(pos_idx), cap(neg_idx)

print("\nangle sweep (fresh runs):", flush=True)
ok = 0
for ang in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
    s = run(ang, thr=THR, sc=SC)
    ps, ns = s[pp].sum(), s[nn].sum()
    act = 1 if ps > ns else 0
    expect = 1 if ang > 0 else 0
    mark = "OK" if act == expect or abs(ang) < 1e-9 else ".."
    if act == expect:
        ok += 1
    print(f"  {ang:+.1f}: +={ps:4.0f} -={ns:4.0f} act={act} {mark}", flush=True)

np.save('subcircuit_motor_pos.npy', pp)
np.save('subcircuit_motor_neg.npy', nn)
np.save('subcircuit_cfg.npy', np.array([THR, SC]))
print(f"saved pools + cfg thr={THR} sc={SC}, sweep_ok={ok}/7", flush=True)
