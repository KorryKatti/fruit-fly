import numpy as np
import scipy.sparse as sp
from sensors import encode_cartpole_state

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]

def run_steps(angle, steps):
    state = np.array([0.0, 0.0, angle, 0.0])
    sensory = encode_cartpole_state(state, n)
    spikes = np.zeros(n, dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)
    hist = []
    for _ in range(steps):
        syn = A @ spikes
        v = 0.95 * v + syn + sensory
        spikes = (v > 2.0).astype(np.float32)
        hist.append((spikes.copy(), v.copy()))
    return hist

# scan steps for best differential
for step_i in [1, 2, 3, 4, 5]:
    hp = run_steps(+0.3, step_i+1)
    hn = run_steps(-0.3, step_i+1)
    sp_p, v_p = hp[step_i]
    sp_n, v_n = hn[step_i]
    diff = v_p - v_n
    diff[0:500] = 0
    nd = np.count_nonzero(np.abs(diff) > 0.01)
    print(f"step {step_i}: |dv|>0.01 in {nd} neurons, max={np.abs(diff).max():.4g}, "
          f"spikes p/n = {sp_p.sum():.0f}/{sp_n.sum():.0f}")

# Use step 5, threshold 0.01
STEP = 5
hp = run_steps(+0.3, STEP+1)
hn = run_steps(-0.3, STEP+1)
_, v_p = hp[STEP]
_, v_n = hn[STEP]
diff = v_p - v_n
diff[0:500] = 0

pos_idx = np.where(diff > 0.01)[0]
neg_idx = np.where(diff < -0.01)[0]
print(f"\nstep {STEP} pools: prefer+ {len(pos_idx)}, prefer- {len(neg_idx)}")

rng = np.random.default_rng(0)
if len(pos_idx) > 500: pos_idx = rng.choice(pos_idx, 500, replace=False)
if len(neg_idx) > 500: neg_idx = rng.choice(neg_idx, 500, replace=False)

print("\nangle   |  +spk  -spk   diff   act |  +v      -v      dv     act")
for ang in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
    h = run_steps(ang, STEP+1)
    sp, v = h[STEP]
    ps, ns = sp[pos_idx].sum(), sp[neg_idx].sum()
    pv, nv = v[pos_idx].mean(), v[neg_idx].mean()
    a1 = 1 if ps > ns else 0
    a2 = 1 if pv > nv else 0
    print(f"{ang:+.1f}  | {ps:5.0f} {ns:5.0f} {ps-ns:6.0f}  {a1}   | {pv:7.2f} {nv:7.2f} {pv-nv:7.2f}  {a2}")

np.save('/tmp/opencode/pos_idx.npy', pos_idx)
np.save('/tmp/opencode/neg_idx.npy', neg_idx)
print(f"\nsaved: +{len(pos_idx)}, -{len(neg_idx)}, step={STEP}")
