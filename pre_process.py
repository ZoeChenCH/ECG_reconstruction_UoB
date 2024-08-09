import csv
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pylab as plt
fs = 100
st= 1*60*fs

def getData(row_file):
    with open(row_file, newline='') as csvfile:
        ECG = []
        PPG = []
        time = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            time.append(['Time'])
            if row['ECG']:
                ECG.append(float(row['ECG']))
            else:
                ECG.append(0.0)
            if row['Pleth']:
                PPG.append(float(row['Pleth']))
            else:
                PPG.append(0.0)

    t = np.arange(len(ECG))/fs
    #ECG_f = butter_bandpass_filter(ECG, 0.1, 25, fs, 5)
    #PPG_f = butter_bandpass_filter(PPG, 0.1, 20, fs, 5)
    ECG_f = ECG[30*fs+1:]
    PPG_f = PPG[30*fs+1:]
    ECG_f = ECG_f
    PPG_f = PPG_f
    return ECG_f, PPG_f

from scipy.signal import butter, filtfilt, resample
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    new_f = 1000
    num_samples_new = int(new_f * len(data) / float(fs))
    signal_resampled = resample(data, num_samples_new)

    nyquist = 0.5 * new_f
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal_resampled)
    y = resample(y, len(data))
    return y

def PPG_findPeaks(PPG_seg):
    PPG_seg = np.array(PPG_seg)
    x = np.arange(len(PPG_seg))/100
    numbers_series = pd.Series(PPG_seg - np.mean(PPG_seg))
    moving_averages = round(numbers_series.ewm(alpha=40/len(PPG_seg), adjust=True).mean(), 5)
    #MA2 = list(map(lambda x: x ** 0.5, moving_averages))
    #PPG_MA = np.array(MA2)
    PPG_MA = pd.Series(moving_averages)
    peaks_MA, _ = find_peaks(PPG_MA, height=np.mean(PPG_MA)*0.5, distance=40)
    peaks_raw, _ = find_peaks(PPG_seg, height=np.mean(PPG_seg), distance=40)


    #plt.plot(x, PPG_seg)
    #plt.plot(x, PPG_MA)
    #plt.plot(x[peaks_MA], PPG_MA[peaks_MA],'o')
    #plt.plot(x[peaks_raw], PPG_seg[peaks_raw], 'x')


    peaks = []
    for peak in peaks_MA:
        if peak >= 20 and peak <= len(PPG_seg)-10:
            peak_ppg = np.argmax(PPG_seg[peak-20:peak+10])
            peak_ppg += peak-20
        elif peak < 20:
            peak_ppg = np.argmax(PPG_seg[0:peak + 10])
        elif peak > len(PPG_seg)-10:
            peak_ppg = np.argmax(PPG_seg[peak - 20:])
            peak_ppg += peak - 20

        #plt.plot(x[peak-20:peak+10], PPG_seg[peak-20:peak+10])
        #plt.plot(x[peak_ppg], PPG_seg[peak_ppg],"ro")
        peaks.append(peak_ppg)

    #plt.show()
    return peaks

def create_segments(data_PPG, data_ECG, segment_length, waveform_length):
    PPG_segments = []
    ECG_segments = []
    PPG_WFset = []
    ECG_WFset = []
    PPG_nWFset = []
    ECG_nWFset = []
    peaks_set = []
    hp_set = []
    he_set = []
    segm_id = []
    id = 0
    x = np.arange(segment_length) / 100
    for i in range(0, len(data_PPG), segment_length):
        if i + segment_length <= len(data_PPG):
            PPG_segment = data_PPG[i:i+segment_length]
            ECG_segment = data_ECG[i:i + segment_length]
            PPG_segments.append(PPG_segment)
            ECG_segments.append(ECG_segment)
            peaks_idx= PPG_findPeaks(PPG_segment)

            if 0:
                x1 = np.arange(len(PPG_segment)) / 100
                plt.figure(figsize=(15, 15))
                plt.subplot(2, 1, 1)
                plt.plot(x1, ECG_segment, 'k', linewidth=3.0)
                plt.xlabel('Time (sec)', fontsize=24)
                plt.title('electrocardiogram (ECG)', fontsize=28)
                plt.ylabel('Amplitude (mV)', fontsize=24)
                plt.subplot(2, 1, 2)
                plt.plot(x1, PPG_segment, 'k', linewidth=3.0)
                plt.xlabel('Time (sec)', fontsize=24)
                plt.title('photoplethysmography  (PPG)', fontsize=28)
                plt.ylabel('Amplitude a.u.', fontsize=24)
                plt.legend(fontsize='16')
                plt.show()

            d_peak = []
            for peak in peaks_idx:
                if peak - int(waveform_length * 0.8) >=0 and peak + int(waveform_length * 0.2) < len(PPG_segment):
                    PPG_WF = PPG_segment[peak - int(waveform_length * 0.8):peak + int(waveform_length * 0.2)]
                    h_PPG = np.max(PPG_WF)-np.min(PPG_WF)
                    PPG_nWF = (PPG_WF-np.min(PPG_WF))/(h_PPG)
                    PPG_nWF = PPG_nWF.astype(np.float32)
                    PPG_WFset.append(PPG_WF)
                    PPG_nWFset.append(PPG_nWF)
                    hp_set.append(h_PPG.astype(np.float32))
                    ECG_WF = ECG_segment[peak - int(waveform_length * 0.8):peak - int(waveform_length * 0.8) + fs]
                    h_ECG = np.max(ECG_WF) - np.min(ECG_WF)
                    ECG_nWF =(ECG_WF-np.min(ECG_WF))/(h_ECG)
                    ECG_nWF = ECG_nWF.astype(np.float32)
                    ECG_WFset.append(ECG_WF)
                    ECG_nWFset.append(ECG_nWF)
                    he_set.append(h_ECG.astype(np.float32))
                    segm_id.append(id)
                    peaks_set.append(peak)

                    '''
                    x1 = np.arange(len(PPG_WF)) / 100
                    x2 = np.arange(len(ECG_WF)) / 100
                    plt.figure(figsize=(15, 15))
                    plt.subplot(2, 1, 1)
                    plt.plot(x2, ECG_WF, 'k', linewidth=3.0)
                    plt.xlabel('Time (sec)', fontsize=24)
                    plt.title('electrocardiogram (ECG)', fontsize=28)
                    plt.ylabel('Amplitude (mV)', fontsize=24)
                    plt.subplot(2, 1, 2)
                    plt.plot(x1, PPG_WF, 'k', linewidth=3.0)
                    plt.xlabel('Time (sec)', fontsize=24)
                    plt.title('photoplethysmography (PPG)', fontsize=28)
                    plt.ylabel('Amplitude a.u.', fontsize=24)
                    plt.legend(fontsize='16')
                    plt.show()
                    '''

                #else:
                    #d_peak.append(peak)
            #peaks_idx = np.setdiff1d(peaks_idx, d_peak)
            #peaks_set.append(peaks_idx)
            id += 1
    return ECG_segments, PPG_segments, PPG_WFset, ECG_WFset, peaks_set, PPG_nWFset, ECG_nWFset, hp_set, he_set, segm_id

def recombine_peaks(peaks, signal, peaks_set,pic):
    x = np.arange(len(signal))/100
    matrix = np.full((len(peaks), len(signal)), np.nan)
    for i in range(0,len(peaks)):
        matrix[i,peaks[i]-int(150*0.8):peaks[i]-int(150*0.8)+len(peaks_set[0])] = peaks_set[i]
    rebuild_signal = np.nanmean(matrix, axis=0)
    rebuild_signal_2 = rebuild_signal.copy()
    avg_peak = np.nanmean(peaks_set, axis=0)
    current_nan_length = 0
    for idx, value in enumerate(rebuild_signal_2):
        if np.isnan(value):
            current_nan_length += 1
            if current_nan_length > 100:
                rebuild_signal_2[idx-100:idx] = avg_peak
                current_nan_length = 0
        else:
            current_nan_length = 0

    rebuild_signal = pd.Series(rebuild_signal)
    rebuild_signal = rebuild_signal.ffill()
    rebuild_signal = rebuild_signal.bfill()
    rebuild_signal = rebuild_signal.interpolate(method='polynomial', order=2).tolist()
    rebuild_signal_2 = pd.Series(rebuild_signal_2)
    rebuild_signal_2 = rebuild_signal_2.ffill()
    rebuild_signal_2 = rebuild_signal_2.bfill()
    rebuild_signal_2 = rebuild_signal_2.interpolate(method='polynomial', order=2).tolist()

    if pic:
        plt.figure()
        plt.plot(x, rebuild_signal, 'k', linewidth = 3, label = 'interpolated signal')
        plt.plot(x, rebuild_signal_2, 'r', linewidth =1, label='add average beat')
        plt.plot(x, signal, 'b--', linewidth=1, label='Actual ECG')
        plt.show()
    return rebuild_signal_2

