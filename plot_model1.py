import matplotlib.pylab as plt
import numpy as np
def plot_model1(x1, ppg, x2, simulated_ecg_segment, ecg):
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 1, 1)
    plt.plot(x1, ppg, 'k', label='Input PPG')
    plt.xlabel('Time (sec)', fontsize=14)
    plt.title('PPG', fontsize=20)
    plt.subplot(2, 1, 2)
    plt.plot(x2, simulated_ecg_segment, label='Simulated ECG')
    plt.plot(x2, ecg - np.min(ecg), label='normalized Actual ECG')
    plt.xlabel('Time (sec)', fontsize=14)
    plt.title('ECG', fontsize=20)
    plt.legend(fontsize='16')
    plt.show()