import time
import numpy as np
import scipy.sparse as sp
from neuron import LIFNeuron
from sensors import encode_cartpole_state

A = sp.load_npz('connectome_adjacency.npz')
n = A.shape[0]
neuron = LIFNeuron(n, decay=0.95, threshold=2.0)
sensory = encode_cartpole_state(np.array([0, 0, 0.1, 0.0]), n)

# warmup
neuron.step(A, sensory)

t0 = time.time()
for _ in range(5):
    neuron.step(A, sensory)
dt = (time.time() - t0) / 5
print(f"neuron.step time: {dt*1000:.1f} ms")

# spike stats in candidate feature region after warmup
for _ in range(10):
    spikes = neuron.step(A, sensory)
region = spikes[8000000:8001000]
print(f"region 8M:8.001M firing rate: {region.mean()*100:.2f}%")

# differential check fresh runs
def run(ang, steps=10):
    nn = LIFNeuron(n, decay=0.95, threshold=2.0)
    s = encode_cartpole_state(np.array([0, 0, ang, 0.0]), n)
    for _ in range(steps):
        sp_ = nn.step(A, s)
    return sp_

sp_p = run(+0.3)
sp_n = run(-0.3)
d = sp_p[8000000:8001000] - sp_n[8000000:8001000]
print(f"diff neurons in 8M region: {np.count_nonzero(d)} / 1000")
d2 = sp_p[500:500000] - sp_n[500:500000]
print(f"diff neurons in 500:500000: {np.count_nonzero(d2)}")
