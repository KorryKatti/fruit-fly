import gymnasium as gym
import numpy as np
import scipy.sparse as sp
import time
from neuron import LIFNeuron
from sensors import encode_cartpole_state
from motors import decode_motor_output

# Load connectome
print("Loading connectome...")
A = sp.load_npz('connectome_adjacency.npz')
n_neurons = A.shape[0]
print(f"Loaded: {n_neurons} neurons, {A.nnz} synapses")

# Initialize neuron
neuron = LIFNeuron(n_neurons, decay=0.95, threshold=10.0)

# CartPole environment
env = gym.make('CartPole-v1')

# Run episodes
n_episodes = 10
max_steps = 500

for episode in range(n_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Sensory encoding (SC=50 matches probe_motor8)
        sensory = encode_cartpole_state(state, n_neurons) * 50.0
        
        # Neural step
        spikes = neuron.step(A, sensory)
        
        # Motor decoding
        action, motor_activity = decode_motor_output(spikes)
        
        # Environment step
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode+1}: {int(episode_reward)} steps")

env.close()
print("Done.")
