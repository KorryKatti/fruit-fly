import numpy as np
import scipy.sparse as sp
from sensors import encode_cartpole_state

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]

# Simulate: encode +angle vs -angle, see which neurons respond differentially
def run(angle, ang_vel=0.0, steps=5):
    state = np.array([0.0, 0.0, angle, ang_vel])
    sensory = encode_cartpole_state(state, n)
    spikes = np.zeros(n, dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)
    decay, thr = 0.95, 2.0
    for _ in range(steps):
        syn = A @ spikes
        v = decay * v + syn + sensory
        spikes = (v > thr).astype(np.float32)
    return spikes, v

print("Running +angle...")
sp_pos, v_pos = run(+0.3)
print("Running -angle...")
sp_neg, v_neg = run(-0.3)

diff = np.abs(v_pos - v_neg)
nz = np.count_nonzero(diff)
print(f"\nvoltage diff neurons: {nz}")
print(f"max |dv|: {diff.max():.4g}")

# candidate: neurons with largest differential response
# exclude sensory pool 0:500
diff[0:500] = 0
cand = np.argsort(-diff)[:522]
print(f"\ntop-522 differential neurons: idx {cand.min()}..{cand.max()}")
print(f"their v_pos mean={v_pos[cand].mean():.4g}, v_neg mean={v_neg[cand].mean():.4g}")

# For each candidate, check if activity is higher for +angle or -angle
higher_for_pos = np.mean(v_pos[cand] > v_neg[cand])
print(f"fraction higher for +angle: {higher_for_pos:.2f}")

# Split into two pools by sign of preference
pref = v_pos[cand] - v_neg[cand]
pos_pool = cand[pref > 0]
neg_pool = cand[pref < 0]
print(f"prefers +angle: {len(pos_pool)}, prefers -angle: {len(neg_pool)}")

# Also check sustained spiking over more steps for a stronger signal
print("\nRunning longer (20 steps)...")
sp_pos2, v_pos2 = run(+0.3, steps=20)
sp_neg2, v_neg2 = run(-0.3, steps=20)
sp_zero, v_zero = run(0.0, steps=20)

# Whole-network spike counts
print(f"spikes @+0.3: {sp_pos2.sum():.0f}, @-0.3: {sp_neg2.sum():.0f}, @0: {sp_zero.sum():.0f}")

# Find pool where spike count differs by sign
# Use in-degree weighted: actually just test candidate pools
for name, pool in [("pos_pref", pos_pool[:261] if len(pos_pool)>=261 else pos_pool),
                   ("neg_pref", neg_pool[:261] if len(neg_pool)>=261 else neg_pool),
                   ("top_diff", cand[:522])]:
    if len(pool) == 0:
        continue
    s_pos = sp_pos2[pool].sum()
    s_neg = sp_neg2[pool].sum()
    s_zero = sp_zero[pool].sum()
    print(f"pool {name} (n={len(pool)}): spikes +ang={s_pos:.0f}, -ang={s_neg:.0f}, 0={s_zero:.0f}")
