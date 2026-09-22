# Balancing CartPole with a fruit fly connectome (mostly)

**NOTE** : this was heavily AI Assisted ( probably obvious ) but i had to know how to do these

Can a chunk of the fly's wiring diagram balance a pole on a cart if I slap a simple readout on top? Kind of — with a bunch of fixes along the way.

**TL;DR:** 50K-neuron subcircuit of MaleCNS + logistic regression → **500/500 steps**. Sensory encoding does most of the work; the connectome part helps close the gap. Sparse subcircuits fail hard.

---

## What this is

A weekend-ish project:

1. Load the male fly CNS connectome (~88M neurons, ~152M synapses)
2. Carve out a smaller subcircuit
3. Simulate it with leaky integrate-and-fire neurons
4. Feed it CartPole state
5. Train a linear readout on the spikes
6. See if the pole stays up

No GPU. No fancy learning on the connectome itself — the wiring is frozen, only the readout is trained.

---

## The boring setup details

**Data:** [MaleCNS v1.0](https://male-cns.janelia.org/) feather file → CSR matrix `A[pre, post] = weight`. Weights are just synapse counts. No E/I labels, no neurotransmitters — raw EM numbers.

**Subcircuit (the one that works):**
- Start from 1000 fake "sensory" neurons + 1000 probed "motor" neurons
- Greedily add neurons that have the most edges into the set (slight bias toward sensory↔motor paths)
- Stop at 50K neurons → **8.8M edges** (avg degree ~176)
- Normalize so each neuron's incoming weights sum to 1 (otherwise everything saturates)

**LIF sim:**
```
v = 0.95 * v + (A.T @ spikes) + sensory
spike if v > 2
reset v to 0
```

`A.T` matters — see [log.md](log.md), I got this backwards for a while.

**Sensory code:** 1000 Gaussian units for `[x, ẋ, θ, θ̇]` (250 each), scaled ×50.

**Readout:** 30 episodes, ε=0.3 random actions for coverage, but **labels are a heuristic expert** (`push right if θ + 0.5·θ̇ > 0`). `LogisticRegression` on spike vectors.

---

## Results

| What the readout sees | Train acc | Eval (mean steps) |
|---|---|---|
| Sensory only (1K) | 97% | 485 |
| Non-sensory only (10K) | 90% | ~100 |
| **Both (11K)** | **100%** | **500** 🎉 |
| Random policy | — | ~20 |

CartPole-v1 "solved" = 500 steps. Combined model hits that consistently (`main_readout.py`).

### Things that mattered

| Change | Before | After |
|---|---|---|
| Sparse 50K subgraph (114K edges) | 26 steps | — |
| Dense 50K subgraph (8.8M edges) | — | 500 (with sensory) |
| Raw weights | 49.7K/50K neurons firing | ~2% firing |
| `A @ spikes` | signals backwards | `A.T @ spikes` fixed it |

### Honest caveat

A lot of the signal is in the **sensory layer** — CartPole's expert is basically `sign(θ)`, and we encode θ directly with RBFs. The connectome neurons refine it (and are required for a perfect score together with sensory), but this isn't "the fly naturally plays CartPole." Non-sensory alone gets ~100 steps, not 500.

---

## Run it

```bash
# assuming subcircuit_A.npz + readout.pkl already exist
uv run python main_readout.py
uv run python main_readout.py --episodes 20 --verbose
```

To rebuild from scratch:

```bash
uv run python extract_subcircuit2.py   # build dense 50K subcircuit
# column-normalize subcircuit_A.npz (see log.md / train_readout history)
uv run python probe_subcircuit.py      # thr/scale sweep + motor pools
uv run python train_readout.py         # collect data, fit readouts, eval
```

**Files that matter:**

| File | What it does |
|---|---|
| `neuron.py` | LIF cells |
| `sensors.py` | CartPole state → 1000 neurons |
| `extract_subcircuit2.py` | the dense extraction that actually works |
| `probe_subcircuit.py` | threshold/scale sweep, motor pool seeds |
| `train_readout.py` | data collection + logistic regression |
| `main_readout.py` | demo / eval loop |
| `log.md` | what broke and what I learned |

Large binaries (`*.feather`, `*.npz`, `readout.pkl`) are gitignored — regenerate or grab from the dataset link.

---

## If I keep going

- Use predicted neurotransmitter types for real inhibition (if/when I wire that up)
- DAgger instead of ε-greedy so train/eval match better
- Try a task flies actually care about (phototaxis, etc.)
- Fit neuron/synapse params instead of hand-tuning thr/scale — cf. Lappalainen et al. 2024

## References (the useful ones)

- MaleCNS: https://male-cns.janelia.org/
- [Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3) — fit parameters on a fixed connectome
- [conn2res](https://github.com/netneurolab/conn2res) — connectome as reservoir + linear readout (same idea, proper library)
- [log.md](log.md) — full failure log

Dataset: https://male-cns.janelia.org/download/#__tabbed_3_3
