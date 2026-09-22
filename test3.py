import numpy as np
from motors import decode_motor_output

# Mock spike output
spikes = np.zeros(88384522)
spikes[88384000:88384522] = np.random.rand(522)  # random motor activity

action, activity = decode_motor_output(spikes)

print(f"Motor activity: {activity:.2f}")
print(f"Action: {action}")
