import gymnasium as gym
import numpy as np
import scipy.sparse as sp
from neuron import LIFNeuron
from sensors import encode_cartpole_state
from motors import decode_motor_output

A = sp.load_npz('connectome_adjacency.npz')
n_neurons = A.shape[0]
neuron = LIFNeuron(n_neurons, decay=0.95, threshold=10.0)
env = gym.make('CartPole-v1')

state, _ = env.reset()

# Run one episode with logging
for step in range(50):
    sensory = encode_cartpole_state(state, n_neurons) * 50.0
    spikes = neuron.step(A, sensory)
    action, motor_activity = decode_motor_output(spikes)

    print(f"Step {step}: angle={state[2]:+.3f}, motor={motor_activity:.0f}, action={action}")

    state, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        break

env.close()
