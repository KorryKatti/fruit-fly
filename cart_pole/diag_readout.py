"""Diagnose: where does expert-action info live in the subcircuit?"""
import numpy as np
import scipy.sparse as sp
import gymnasium as gym
from sklearn.linear_model import LogisticRegression
from neuron import LIFNeuron
from sensors import encode_cartpole_state

z = np.load('subcircuit_A.npz')
A = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
n = A.shape[0]
sensory_map = np.load('subcircuit_sensory.npy')
THR, SC = 5.0, 20.0

rng = np.random.default_rng(0)
nonsensory = np.setdiff1d(np.arange(n), sensory_map)
feat10k = np.sort(rng.choice(nonsensory, 10_000, replace=False))
feat_all_ns = nonsensory

def encode_sub(state):
    s = np.zeros(n, dtype=np.float32)
    s[sensory_map] = encode_cartpole_state(state, 1000)
    return s * SC

def expert(state):
    return 1 if (state[2] + 0.5 * state[3]) > 0 else 0

# collect richer traces: spikes, voltage, history
env = gym.make('CartPole-v1')
HIST = 5
rows = []  # dict per step
for ep in range(20):
    state, _ = env.reset(seed=ep)
    neuron = LIFNeuron(n, decay=0.95, threshold=THR)
    hist = []
    for step in range(400):
        sens = encode_sub(state)
        spikes = neuron.step(A, sens)
        hist.append(spikes.copy())
        if len(hist) > HIST:
            hist.pop(0)
        spike_sum = np.sum(hist, axis=0)
        rows.append({
            'spikes': spikes[feat10k],
            'spike_sum': spike_sum[feat10k],
            'volt': neuron.v[feat10k],
            'sens': spikes[sensory_map],
            'sens_sum': spike_sum[sensory_map],
            'state': state.copy(),
            'y': expert(state),
            'volt_all_ns': None,
        })
        # full nonsensory voltage too heavy every step — sample every step but only 50k ok
        rows[-1]['volt_ns_all'] = neuron.v[feat_all_ns]
        rows[-1]['spike_ns_all'] = spikes[feat_all_ns]
        action = rows[-1]['y'] if rng.random() > 0.3 else int(rng.integers(2))
        state, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
env.close()

y = np.array([r['y'] for r in rows])
print(f"samples={len(y)} balance={y.mean():.3f}", flush=True)

def eval_feat(name, X):
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"  {name:24s} train_acc={acc:.3f}", flush=True)
    return acc

# raw state (linear boundary = expert itself)
eval_feat("state [x,xd,th,thd]", np.stack([r['state'] for r in rows]))
eval_feat("state [th,thd]", np.stack([[r['state'][2], r['state'][3]] for r in rows]))

# sensory spikes direct
eval_feat("sensory spikes (1000)", np.stack([r['sens'] for r in rows]))
eval_feat("sensory spike_sum", np.stack([r['sens_sum'] for r in rows]))

# non-sensory
eval_feat("10k spikes", np.stack([r['spikes'] for r in rows]))
eval_feat("10k spike_sum5", np.stack([r['spike_sum'] for r in rows]))
eval_feat("10k voltage", np.stack([r['volt'] for r in rows]))
eval_feat("all_ns spikes", np.stack([r['spike_ns_all'] for r in rows]))
eval_feat("all_ns spike_sum5 via 10k...", np.stack([r['spike_sum'] for r in rows]))

# also all_ns voltage - do it
eval_feat("all_ns voltage", np.stack([r['volt_all_ns'] for r in rows]))

# spike_sum on all nonsensory - need to store; approximate with volt history mean not available
# recompute quickly from stored spike_sum only on 10k; skip

print("done", flush=True)
