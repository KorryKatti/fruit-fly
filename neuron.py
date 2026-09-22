import numpy as np
import scipy.sparse as sp

class LIFNeuron:
    """Leaky Integrate and Fire neuron model"""
    """Fancy words that i don't knwo the meaning of"""
    def __init__(self,n_neurons,decay=0.95,threshold=1.0,reset=0.0):
        self.n_neurons = n_neurons
        self.decay = decay
        self.threshold = threshold
        self.reset = reset

        # state
        self.v = np.zeros(n_neurons,dtype=np.float32)#membrane voltage
        self.spikes=np.zeros(n_neurons,dtype=np.float32) # spike output
            
    def step(self,A,sensory_input):
        """
        One timestep of neural dynamics
        Args:
            A: adjacency matrix (sparse,shape(n_neurons,n_neurons)
            sensory_input: external input (shape (n_neurons,))

        Returns:
                                    spikes:binary spike output (shape (n_neurons,))
        """
        # synaptic input from previous spikes
        synaptic = A @ self.spikes

                                 #integrate : decay volate + synaptic input + sensory
        self.v = self.decay*self.v+synaptic+sensory_input

        #threshold : spike if voltage > threshold
        self.spikes = (self.v>self.threshold).astype(np.float32)

        # resset voltage after spike
        return self.spikes
