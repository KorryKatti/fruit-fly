import gymnasium as gym
from sensors import encode_cartpole_state
import numpy as np

env = gym.make('CartPole-v1')
state, _ = env.reset()

sensory = encode_cartpole_state(state, n_neurons=88384522)

print(f"State: {state}")
print(f"Sensory input shape: {sensory.shape}")
print(f"Sensory input sum: {sensory.sum()}")
print(f"Sensory nonzero: {np.count_nonzero(sensory)}")
