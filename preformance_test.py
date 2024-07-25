import numpy as np
def average_difference(sim_signal, actual_signal):
    return np.mean(abs(sim_signal - actual_signal))