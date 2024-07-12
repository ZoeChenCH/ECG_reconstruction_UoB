import numpy as np
import matplotlib.pylab as plt
import csv
from scipy.signal import find_peaks, butter, lfilter
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

row_file = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case01/fulldata/uq_vsd_case01_fulldata_02.csv')
#row_file = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case02/uq_vsd_case02_fulldata_01.csv')
ECG = []
PPG = []
time = []
fs = 100

with open(row_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        time.append(row['Time'])
        if row['ECG']:
            ECG.append(float(row['ECG']))
        else:
            ECG.append(0.0)
        if row['Pleth']:
            PPG.append(float(row['Pleth']))
        else:
            PPG.append(0.0)

t = np.arange(len(ECG))/fs

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#ECG_f = butter_bandpass_filter(ECG, 0.1, 25, fs, 5)
#PPG_f = butter_bandpass_filter(PPG, 0.1, 20, fs, 5)
ECG_f = ECG[30*fs+1:]
PPG_f = PPG[30*fs+1:]

st= 1*60*fs
dur = 10 * fs
wl = fs*1.5

def create_segments(data_PPG, data_ECG, segment_length=dur, waveform_length = wl):
    PPG_segments = []
    ECG_segments = []
    PPG_WFset = []
    ECG_WFset = []
    peaks_set = []
    for i in range(0, len(data_PPG), segment_length):
        if i + segment_length <= len(data_PPG):
            PPG_segment = data_PPG[i:i+segment_length]
            ECG_segment = data_ECG[i:i + segment_length]
            PPG_segments.append(PPG_segment)
            ECG_segments.append(ECG_segment)

            peaks_idx, _ = find_peaks(PPG_segment, height = np.quantile(PPG_segment,0.50))

            for peak in peaks_idx:
                if peak - int(waveform_length * 0.8) >=0 and peak + int(waveform_length * 0.2) < len(PPG_segment):
                    PPG_WF = PPG_segment[peak - int(waveform_length * 0.8):peak + int(waveform_length * 0.2)]
                    PPG_WFset.append(PPG_WF)
                    ECG_WF = ECG_segment[peak - int(waveform_length * 0.8):peak - int(waveform_length * 0.8)+fs]
                    ECG_WFset.append(ECG_WF)
                    peaks_set.append(peaks_idx+i)
    return ECG_segments, PPG_segments, PPG_WFset, ECG_WFset, peaks_set

ECG_segments, PPG_segments, PPG_WFset, ECG_WFset, peaks_set = create_segments(PPG_f, ECG_f, dur, wl)


'''
for i in range(26,50):
    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(np.arange(len(ECG_segments[i]))/fs, ECG_segments[i])
    axs[0].set_ylabel('ECG', fontsize=14)
    axs[1].set_xlabel('Time (s)', fontsize=14)
    axs[0].grid(True)
    axs[1].plot(np.arange(len(PPG_segments[i]))/fs, PPG_segments[i])
    #axs[1].plot(peaks_set[4], PPG_segments[peaks_set[4]], color = 'red')
    axs[1].set_ylabel('PPG', fontsize = 14)
    axs[1].set_xlabel('Time (s)', fontsize=14)
    axs[1].grid(True)
    plt.show()
'''
"""for i in range(10):
    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(ECG_WFset[i])
    axs[0].set_ylabel('ECG')
    axs[0].grid(True)
    axs[1].plot(PPG_WFset[i])
    axs[1].set_ylabel('PPG')
    axs[1].set_xlabel('Time (s)')
    axs[1].grid(True)
    plt.show()
"""
from sklearn.model_selection import train_test_split
PPG_WFset = np.array(PPG_WFset).astype(np.float32)
ECG_WFset = np.array(ECG_WFset).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(PPG_WFset, ECG_WFset, test_size=0.2, random_state=42)

import jax.numpy as jnp
import jax
from jax import random
import optax
from flax import linen as nn
from flax.training import train_state

X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

class ECGModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu,
        x = nn.Dense(features=100)(x)
        return x

class TrainState(train_state.TrainState):
    batch_stats: jnp.ndarray

def compute_loss(params, batch):
    inputs, targets = batch
    predictions = model.apply({'params': params}, inputs)
    loss = jnp.mean((predictions - targets) ** 2 )
    return loss

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        loss = compute_loss(params, batch)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

rng = jax.random.PRNGKey(0)
model = ECGModel()
params = model.init(rng, jnp.ones([1, int(wl)]))['params']
tx = optax.adam(learning_rate=0.001)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

epochs = 1000
batch_size = 32
for epoch in range(epochs):
    batch_indices = np.random.choice(len(X_train), batch_size)
    batch = (X_train[batch_indices], y_train[batch_indices])
    state = train_step(state, batch)

    if epoch % 100 == 0:
        current_loss = compute_loss(state.params, batch)
        print(f'Epoch {epoch}, Loss: {current_loss:}')

import matplotlib.pyplot as plt
predictions = model.apply({'params': state.params}, X_test)

index = np.random.randint(0, len(X_test))
plt.figure(figsize=(12, 12))
plt.plot(y_test[index], label='Actual ECG')
plt.plot(predictions[index], label='Predicted ECG')
plt.legend()
plt.show()