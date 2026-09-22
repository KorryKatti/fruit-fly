import numpy as np

def encode_cartpole_state(state, n_neurons, n_sensory=1000):
    """
    Map cartpole state to neural input

    Args:
        state: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        n_neurons: total neurons in network
        n_sensory: number of neurons dedicated to sensing (4 x 250)

    Returns:
        sensory_input: array of shape (n_neurons,)
    """
    sensory = np.zeros(n_neurons, dtype=np.float32)

    # pole angle (neurons 0-249): tuned to angles -pi to +pi
    angle = state[2]
    angle_centers = np.linspace(-np.pi, np.pi, 250)
    sensory[0:250] = np.exp(-((angle_centers - angle)**2) / 0.5)

    # pole angular velocity (neurons 250-499): tuned to velocities -2 to +2
    ang_vel = state[3]
    vel_centers = np.linspace(-2, 2, 250)
    sensory[250:500] = np.exp(-((vel_centers - ang_vel)**2) / 0.5)

    # cart position (neurons 500-749): tuned to positions -2.4 to +2.4
    cart_pos = state[0]
    pos_centers = np.linspace(-2.4, 2.4, 250)
    sensory[500:750] = np.exp(-((pos_centers - cart_pos)**2) / 0.5)

    # cart velocity (neurons 750-999): tuned to velocities -3 to +3
    cart_vel = state[1]
    cvel_centers = np.linspace(-3, 3, 250)
    sensory[750:1000] = np.exp(-((cvel_centers - cart_vel)**2) / 0.5)

    return sensory
