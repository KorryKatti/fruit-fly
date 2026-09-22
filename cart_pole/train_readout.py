"""Option B on E2 dense subcircuit: readout with feature ablations."""
import pickle
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
cfg = np.load('subcircuit_cfg.npy')
THR, SC = float(cfg[0]), float(cfg[1])
print(f"sub_A {A.shape} nnz={A.nnz} thr={THR} sc={SC}", flush=True)

rng = np.random.default_rng(0)
nonsensory = np.setdiff1d(np.arange(n), sensory_map)
FEAT_NS = np.sort(rng.choice(nonsensory, size=min(10_000, nonsensory.size), replace=False))
FEAT_SENS = sensory_map.copy()
FEAT_BOTH = np.sort(np.concatenate([FEAT_NS, FEAT_SENS]))
print(f"features: ns={FEAT_NS.size} sens={FEAT_SENS.size} both={FEAT_BOTH.size}", flush=True)

def encode_sub(state):
    s = np.zeros(n, dtype=np.float32)
    s[sensory_map] = encode_cartpole_state(state, 1000)
    return s * SC

def expert(state):
    return 1 if (state[2] + 0.5 * state[3]) > 0 else 0

# collect
N_EP, EPS, MAX_STEPS = 30, 0.3, 500
env = gym.make('CartPole-v1')
X_ns, X_sens, X_both, y_list = [], [], [], []
collect_rng = np.random.default_rng(1)

for ep in range(N_EPISODES := N_EP):
    state, _ = env.reset(seed=int(collect_rng.integers(1e9)))
    neuron = LIFNeuron(n, decay=0.95, threshold=THR)
    for step in range(MAX_STEPS):
        spikes = neuron.step(A, encode_sub(state))
        label = expert(state)
        action = int(collect_rng.integers(2)) if collect_rng.random() < EPS else label
        X_ns.append(spikes[FEAT_NS])
        X_sens.append(spikes[FEAT_SENS])
        X_both.append(spikes[FEAT_BOTH])
        y_list.append(label)
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if (ep + 1) % 10 == 0:
        print(f"  {ep+1} eps, {len(y_list)} samples", flush=True)
env.close()

y = np.array(y_list, dtype=np.int16)
X_ns = np.stack(X_ns)
X_sens = np.stack(X_sens)
X_both = np.stack(X_both)
print(f"samples={len(y)} balance={y.mean():.3f} ns_sparse={X_ns.mean():.3f} sens_sparse={X_sens.mean():.3f}", flush=True)

def fit_eval(name, X):
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"  {name:20s} train_acc={acc:.3f}", flush=True)
    return clf, acc

print("train:", flush=True)
clf_ns, acc_ns = fit_eval("10k non-sensory", X_ns)
clf_sens, acc_sens = fit_eval("sensory only", X_sens)
clf_both, acc_both = fit_eval("sensory+ns", X_both)

# pick best non-sensory-only for the "honest" connectome claim; also eval both
def evaluate(clf, feats, tag, n_ep=10):
    env = gym.make('CartPole-v1')
    rewards = []
    for ep in range(n_ep):
        state, _ = env.reset(seed=1000 + ep)
        neuron = LIFNeuron(n, decay=0.95, threshold=THR)
        total = 0
        for step in range(500):
            spikes = neuron.step(A, encode_sub(state))
            action = int(clf.predict(spikes[feats].reshape(1, -1))[0])
            state, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rewards.append(total)
    env.close()
    print(f"  {tag:20s} mean={np.mean(rewards):.1f} +/- {np.std(rewards):.1f} "
          f"min={np.min(rewards):.0f} max={np.max(rewards):.0f}", flush=True)
    return rewards

print("eval:", flush=True)
evaluate(clf_ns, FEAT_NS, "10k non-sensory")
evaluate(clf_sens, FEAT_SENS, "sensory only")
evaluate(clf_both, FEAT_BOTH, "sensory+ns")

with open('readout.pkl', 'wb') as f:
    pickle.dump({
        'clf_ns': clf_ns, 'clf_sens': clf_sens, 'clf_both': clf_both,
        'feat_ns': FEAT_NS, 'feat_sens': FEAT_SENS, 'feat_both': FEAT_BOTH,
        'thr': THR, 'sc': SC,
    }, f)
print("saved readout.pkl", flush=True)
