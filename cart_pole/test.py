import scipy.sparse as sp
import numpy as np
import time
from neuron import LIFNeuron

A = sp.load_npz('connectome_adjacency.npz')
n_neurons = A.shape[0]

neuron = LIFNeuron(n_neurons, decay=0.95, threshold=2.0)

sensory = np.random.randn(n_neurons).astype(np.float32) * 1.0

start = time.time()
spikes = neuron.step(A, sensory)
elapsed = time.time() - start

print(f"Spikes fired: {spikes.sum()}")
print(f"Time: {elapsed:.3f}s")
print(f"Firing rate: {spikes.sum() / n_neurons * 100:.2f}%")
