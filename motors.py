import numpy as np
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_pos = np.load(os.path.join(_DIR, 'data', 'motor_pos_idx.npy'))
_neg = np.load(os.path.join(_DIR, 'data', 'motor_neg_idx.npy'))

def decode_motor_output(spikes):
    """
    Differential motor readout.
    pos pool prefers +pole angle, neg pool prefers -pole angle.
    action 1 (push right) when pos pool fires more, else 0 (push left).
    """
    pos_act = spikes[_pos].sum()
    neg_act = spikes[_neg].sum()
    action = 1 if pos_act > neg_act else 0
    return action, float(pos_act - neg_act)
