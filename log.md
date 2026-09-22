# Log — stuff I learned the hard way

Casual notes from trying to get a fly connectome subcircuit to balance CartPole. Failures kept in on purpose.

---

## 1. The weights are just counts

`weight` in the feather file = how many synapse segments EM found. Range 1–2591. No GABA vs glutamate, no receptors, no plasticity. Using these raw as LIF gains is going to cause problems (spoiler: it did).

---

## 2. I had the matrix backwards

Saved as `A[pre, post] = weight` (double-checked against the feather: row→col is always pre→post).

I was doing:

```python
synaptic = A @ spikes   # Σ_j w(i→j) * spikes[j]  — uses i's OUTPUTS, wrong
```

Should be:

```python
synaptic = A.T @ spikes  # Σ_j w(j→i) * spikes[j]  — inputs TO i, correct
```

Everything before this fix was running the connectome in reverse. Motor signals were garbage/noisy. Classic "why doesn't anything work" bug.

---

## 3. Forgot half the state

Only encoded pole angle + angular velocity at first. CartPole also has cart position and velocity — pole can be balanced while the cart rolls off the map.

Now: 4 × 250 RBF units = 1000 sensory neurons.

---

## 4. Everything fires all the time

Big synapse (2591) + threshold 2 → spike spike spike. Saw **49,700 / 50,000** neurons active on the dense subcircuit. Useless for a readout.

**Fix:** normalize each neuron's incoming weights to sum to 1. Firing dropped to ~2%. Stable.

Also: I ran threshold/scale sweeps in probe scripts, found good values, and **forgot to put them in `main.py`**. If you tune something, write it back into the script you actually run.

---

## 5. Indices 0–999 aren't real sensory neurons

They're just low sorted body IDs that I overwrite with a CartPole code. Same deal for "motor" pools — they're neurons that happened to respond in a probe, not identified motor neurons.

Fine for a demo. Don't write "the fly's motor cortex" in a paper.

---

## 6. Looks great open-loop, fails closed-loop

Fresh sim at fixed angle ±0.3 → beautiful monotonic motor pools.

Same decoder in a real episode → sign flips, oscillation, dies at step 10.

Live episodes have: network state carrying over, cart moving, bursting dynamics. Fitting on static angles doesn't transfer.

**Lesson:** train and eval the readout on actual closed-loop rollouts.

---

## 7. First readout idea was nonsense

I sketched:

```python
action = env.action_space.sample()  # "explore"
y.append(action)                    # label = random noise
```

Spikes depend on state. Random actions don't. Logistic regression learns a coin flip (~50%).

**Fix:** explore with ε=0.3, but label every step with a dumb expert:

```python
y = 1 if angle + 0.5 * ang_vel > 0 else 0
```

---

## 8. `spikes[8000000:8001000]` was empty

That slice: 0.8% firing, **1 out of 1000** neurons told +0.3 apart from −0.3.

Reading noise. Always check that your features actually vary with the thing you care about before fitting anything.

---

## 9. Full 88M sim is painfully slow

~**2.4 seconds per step**. A "quick" 100×200 collection run ≈ half a day. Not a hobbyist-friendly loop.

**Fix:** cut out a 50K subcircuit. Steps become milliseconds.

---

## 10. Sparse subgraph = dead subgraph (biggest surprise)

First extraction: one-hop neighbors of seeds → 50K neurons but only **114K edges** (~2 per neuron). Induced subgraph drops any edge whose other end wasn't kept.

| | Sparse (E1) | Dense (E2) |
|---|---|---|
| Edges | 114K | **8.8M** |
| Avg degree | ~2 | **176** |
| Non-sensory train acc | 55% (chance) | **90%** |
| Eval (non-sensory) | 26 steps | ~100 |
| Eval (sensory+ns) | fail | **500** |

Second try: greedily add the neuron with the most edges into the set already (tiny bonus if it's on a 4-hop sensory↔motor path). Same 50K nodes, **77× more edges**.

Turns out the problem wasn't "CartPole is too alien for a fly" — it was **"I threw away all the paths."**

---

## 11. What the ablation honestly says

- Sensory-only readout: **485** steps  
- Non-sensory only: **~100**  
- Both: **500**

The expert is basically `sign(θ)`, and we encode θ with RBFs — of course the input layer is strong. The 50K fly-ish neurons are a weak but real feature layer: they don't solve it alone, but with sensory they close 485 → 500.

**OK to say:** dense connectome subcircuit + linear readout solves CartPole; most signal is task-aligned sensory.  
**Not OK to say:** the fly connectome natively implements CartPole.

---

## 12. What other people did (so I don't reinvent poorly)

- **Lappalainen et al. 2024** — keep connectome fixed, **fit** neuron/synapse free params on a task. Raw counts aren't dynamics.
- **conn2res** — library for exactly "frozen connectome + trained linear readout."
- **FLM** — frozen MaleCNS + trained readout (closest cousin to this repo).
- **flyGNN** — whole FlyWire net controlling a physics fly.
- **Shiu et al. 2024** — whole brain sim that predicts what fires when you poke sensory neurons.

Nobody seems to have done MaleCNS → CartPole. Small pond.

---

## 13. Dead ends not worth revisiting (unless something changes)

- Hand-tuned differential motor decoder as the main controller — too brittle live
- Random actions as supervised labels
- Reading spikes from silent slices
- Hyperparameter search on the full 88M graph
- Python-loop normalizing 88M columns (never finishes — normalize the subcircuit)
- Quitting early with "flies didn't evolve for this" while basic bugs (direction, saturation, sparsity) were still in the code

---

## 14. What actually works end-to-end

```
feather → connectome_adjacency.npz     # once
extract_subcircuit2.py                 # 50K, greedy, path-biased
normalize columns of subcircuit_A.npz  # incoming sum = 1
probe_subcircuit.py                    # thr/scale + motor seeds
train_readout.py                       # ε=0.3, expert labels, ablations
main_readout.py                        # watch it balance
```

---

## 15. Next ideas (in roughly the order I'd try them)

1. Wire in predicted neurotransmitter signs if I can get them → actual inhibition  
2. DAgger so the readout sees states it actually visits at eval  
3. A task flies care about (phototaxis / optic flow) with real cell types  
4. Fit LIF/synaptic params instead of hand-tuning thr and scale  
5. Keep reporting non-sensory-only numbers so I don't fool myself
