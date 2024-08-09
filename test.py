import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

fs = 1000
duration = 5

t = np.linspace(0, duration, int(fs * duration), endpoint=False)

signal = (np.sin(2 * np.pi * 10 * t) +  # 1 Hz 成分
          np.sin(2 * np.pi * 100 * t))

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    print(low, high, fs, order)
    return y

lowcut = 0.1
highcut = 40.0

filtered_signal = bandpass_filter(signal, lowcut, highcut, fs, order=4)

plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal with Multiple Frequencies')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, filtered_signal)
plt.title(f'Filtered Signal (Bandpass {lowcut}-{highcut} Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, signal, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal', linestyle='--')
plt.title('Comparison of Original and Filtered Signals')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()