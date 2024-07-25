import csv
import numpy as np
from scipy.signal import butter, lfilter
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

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
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
    x = np.arange(segment_length) / 100
    for i in range(0, len(data_PPG), segment_length):
        if i + segment_length <= len(data_PPG):
            PPG_segment = data_PPG[i:i+segment_length]
            ECG_segment = data_ECG[i:i + segment_length]
            PPG_segments.append(PPG_segment)
            ECG_segments.append(ECG_segment)
            peaks_idx= PPG_findPeaks(PPG_segment)

            PPG_oneBeat = []
            ECG_oneBeat = []
            PPG_oneBeatn = []
            ECG_oneBeatn = []
            d_peak = []
            for peak in peaks_idx:
                if peak - int(waveform_length * 0.8) >=0 and peak + int(waveform_length * 0.2) < len(PPG_segment):
                    PPG_WF = PPG_segment[peak - int(waveform_length * 0.8):peak + int(waveform_length * 0.2)]
                    h_PPG = np.max(PPG_WF)-np.min(PPG_WF)
                    PPG_nWF = (PPG_WF-np.min(PPG_WF))/(h_PPG)
                    PPG_nWF = PPG_nWF.astype(np.float32)
                    PPG_oneBeat.append(PPG_WF)
                    PPG_oneBeatn.append(PPG_nWF)
                    hp_set.append(h_PPG.astype(np.float32))
                    ECG_WF = ECG_segment[peak - int(waveform_length * 0.8):peak - int(waveform_length * 0.8)+fs]
                    h_ECG = np.max(ECG_WF) - np.min(ECG_WF)
                    ECG_nWF =(ECG_WF-np.min(ECG_WF))/(h_ECG)
                    ECG_nWF = ECG_nWF.astype(np.float32)
                    ECG_oneBeat.append(ECG_WF)
                    ECG_oneBeatn.append(ECG_nWF)
                    he_set.append(h_ECG.astype(np.float32))

                else:
                    d_peak.append(peak)
            peaks_idx = np.setdiff1d(peaks_idx, d_peak)
            peaks_set.append(peaks_idx)
            PPG_WFset.append(PPG_oneBeat)
            ECG_WFset.append(ECG_oneBeat)
            PPG_nWFset.append(PPG_oneBeatn)
            ECG_nWFset.append(ECG_oneBeatn)
    return ECG_segments, PPG_segments, PPG_WFset, ECG_WFset, peaks_set, PPG_nWFset, ECG_nWFset, hp_set, he_set

def recombine_peaks(peaks, signal, peaks_set):
    x = np.arange(len(signal))/100
    matrix = np.full((len(peaks), len(signal)), np.nan)
    for i in range(0,len(peaks)):
        matrix[i,peaks[i]-int(150*0.8):peaks[i]+int(150*0.2)] = peaks_set[i]
    rebuild_signal = np.nanmean(matrix, axis=0)
    fig = plt.figure()
    plt.plot(x, signal)
    plt.plot(x[peaks], signal[peaks],'o')
    plt.plot(x, rebuild_signal)
    plt.show()
    return rebuild_signal