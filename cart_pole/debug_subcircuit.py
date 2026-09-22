"""Debug one CartPole episode on the extracted subcircuit (fast)."""
import gymnasium as gym
import numpy as np
import scipy.sparse as sp
from neuron import LIFNeuron
from sensors import encode_cartpole_state

# load sub A from npz archive
z = np.load('subcircuit_A.npz')
A = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
n = A.shape[0]
sensory_map = np.load('subcircuit_sensory.npy')  # [i] = new idx of old sensory neuron i
pos = np.load('subcircuit_motor_pos.npy')
neg = np.load('subcircuit_motor_neg.npy')
print(f"sub_A {A.shape} nnz={A.nnz}, sensory={sensory_map.size}, motors={pos.size}/{neg.size}")

# quick path check: BFS forward from sensory, see if any motor reached
from collections import deque
frontier = set(sensory_map.tolist())
seen = np.zeros(n, dtype=bool)
seen[sensory_map] = True
reached_motor = 0
for hop in range(8):
    nxt = set()
    for u in frontier:
        s, e = A.indptr[u], A.indptr[u + 1]
        for v in A.indices[s:e]:
            if not seen[v]:
                seen[v] = True
                nxt.add(int(v))
    frontier = nxt
    hit = sum(1 for m in np.concatenate([pos, neg]) if seen[m])
    print(f"  hop {hop+1}: new={len(frontier)} seen={seen.sum()} motor_reached={hit}/1000")
    if not frontier:
        break

def encode_sub(state, n_sub):
    full = encode_cartpole_state(state, 1000)  # values for old 0:1000
    sensory = np.zeros(n_sub, dtype=np.float32)
    sensory[sensory_map] = full
    return sensory

def decode(spikes):
    pa, na = spikes[pos].sum(), spikes[neg].sum()
    return (1 if pa > na else 0), float(pa - na)

thr, sc = 5.0, 20.0
try:
    cfg = np.load('subcircuit_cfg.npy')
    thr, sc = float(cfg[0]), float(cfg[1])
except Exception:
    pass
print(f"cfg thr={thr} sc={sc}")
neuron = LIFNeuron(n, decay=0.95, threshold=thr)
env = gym.make('CartPole-v1')
state, _ = env.reset()

for step in range(50):
    sensory = encode_sub(state, n) * sc
    spikes = neuron.step(A, sensory)
    action, motor = decode(spikes)
    print(f"Step {step}: x={state[0]:+.2f} th={state[2]:+.3f} motor={motor:+.0f} act={action} "
          f"spikes={int(spikes.sum())}")
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

print(f"died/success at step {step}, reward~{step+1}")
env.close()
