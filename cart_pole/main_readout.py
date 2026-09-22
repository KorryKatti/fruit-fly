"""Run CartPole with connectome subcircuit + fitted linear readout.

Prereqs (already in repo):
  subcircuit_A.npz, subcircuit_sensory.npy, subcircuit_cfg.npy, readout.pkl

Usage:
  uv run python main_readout.py
  uv run python main_readout.py --episodes 20
"""
import argparse
import pickle
import numpy as np
import scipy.sparse as sp
import gymnasium as gym
from neuron import LIFNeuron
from sensors import encode_cartpole_state


def load_subcircuit():
    z = np.load('subcircuit_A.npz')
    A = sp.csr_matrix(
        (z['data'], z['indices'], z['indptr']),
        shape=tuple(z['shape']),
    )
    sensory_map = np.load('subcircuit_sensory.npy')
    cfg = np.load('subcircuit_cfg.npy')
    return A, sensory_map, float(cfg[0]), float(cfg[1])


def load_readout(path='readout.pkl'):
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    # prefer combined model (best in eval), fall back sens-only then ns-only
    if bundle.get('clf_both') is not None:
        return bundle['clf_both'], bundle['feat_both']
    if bundle.get('clf_sens') is not None:
        return bundle['clf_sens'], bundle['feat_sens']
    return bundle['model'], bundle['features']


def encode_sub(state, sensory_map, n, sc):
    s = np.zeros(n, dtype=np.float32)
    s[sensory_map] = encode_cartpole_state(state, 1000)
    return s * sc


def main():
    parser = argparse.ArgumentParser(description='CartPole via connectome readout')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--readout', default='readout.pkl')
    parser.add_argument('--verbose', action='store_true', help='per-step logging')
    args = parser.parse_args()

    print('Loading subcircuit...', flush=True)
    A, sensory_map, thr, sc = load_subcircuit()
    n = A.shape[0]
    print(f'  A {A.shape} nnz={A.nnz} thr={thr} sc={sc}', flush=True)

    print(f'Loading readout ({args.readout})...', flush=True)
    clf, feats = load_readout(args.readout)
    print(f'  features: {feats.size} neurons', flush=True)

    env = gym.make('CartPole-v1')
    rewards = []

    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        neuron = LIFNeuron(n, decay=0.95, threshold=thr)
        total = 0.0

        for step in range(args.max_steps):
            spikes = neuron.step(A, encode_sub(state, sensory_map, n, sc))
            action = int(clf.predict(spikes[feats].reshape(1, -1))[0])
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward

            if args.verbose:
                print(
                    f'  ep {ep} step {step}: th={state[2]:+.3f} '
                    f'act={action} spikes={int(spikes.sum())}',
                    flush=True,
                )

            if terminated or truncated:
                break

        rewards.append(total)
        mark = ' SOLVED' if total >= args.max_steps else ''
        print(f'Episode {ep + 1}: {int(total)} steps{mark}', flush=True)

    env.close()

    rewards = np.asarray(rewards, dtype=np.float64)
    print(
        f'\nmean={rewards.mean():.1f} +/- {rewards.std():.1f} '
        f'min={rewards.min():.0f} max={rewards.max():.0f} '
        f'(random~20, solved=500)',
        flush=True,
    )
    return 0 if rewards.mean() >= 100 else 1


if __name__ == '__main__':
    raise SystemExit(main())
