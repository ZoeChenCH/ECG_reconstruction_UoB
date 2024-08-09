import numpy as np
import matplotlib.pylab as plt
#testtestestest
row_file1 = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case01/fulldata/uq_vsd_case01_fulldata_02.csv')
row_file3 = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case02/uq_vsd_case02_fulldata_01.csv')
row_file2 = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case01/fulldata/uq_vsd_case01_fulldata_03.csv')
row_file = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case14/uq_vsd_case14_fulldata_03.csv')
fs = 100
dur = 7 * fs
wl = fs* 1.5

from pre_process import getData, create_segments, recombine_peaks
ECG1, PPG1 = getData(row_file1)
#ECG2, PPG2 = getData(row_file)
#ECG_test, PPG_test = getData(row_file1)

ECG_segments1, PPG_segments1, PPG_WFset1, ECG_WFset1, peaks_set1, PPG_nWFset1, ECG_nWFset1, hp_set1, he_set1, segm_id1= create_segments(PPG1, ECG1, dur, wl)
#ECG_segments2, PPG_segments2, PPG_WFset2, ECG_WFset2, peaks_set2, PPG_nWFset2, ECG_nWFset2, hp_set2, he_set2, segm_id2= create_segments(PPG2, ECG2, dur, wl)
#ECG_segmentsT, PPG_segmentsT, PPG_WFsetT, ECG_WFsetT, peaks_setT, PPG_nWFsetT, ECG_nWFsetT, hp_setT, he_setT, segm_idT= create_segments(PPG_test, ECG_test, dur, wl)
'''
segments: 10 second period
WFset: waveform of beats
peaks_set: PPG peaks location in each segment
nWFset: normalized waveform
segm_id: the segment that this beat belong to
'''
ECG_segments = ECG_segments1
PPG_segments = PPG_segments1
PPG_WFset = PPG_WFset1
ECG_WFset = ECG_WFset1
peaks_set = peaks_set1
PPG_nWFset = PPG_nWFset1
ECG_nWFset = ECG_nWFset1
hp_set = hp_set1
he_set = he_set1
segm_id = segm_id1

# ppg_temp = np.array(PPG_segments1[0])
# x = np.arange(len(ppg_temp))/100
# ll = [index for index, value in enumerate(segm_id1) if value == 0]
# peaks_input = [PPG_WFset1[i] for i in ll]
# p = [peaks_set1[u] for u in ll]
# new_ppg = recombine_peaks(p, ppg_temp, peaks_input)

from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from CNNmodel import CNNModel,PPGtoECGDataset

#dataset = PPGtoECGDataset(PPG_segments, ECG_segments)
dataset = PPGtoECGDataset(PPG_nWFset, ECG_nWFset)
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
indices = list(range(len(dataset)))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

model = CNNModel()
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(),lr=0.005)

num_epochs = 100
loss_set = []
for epoch in range(num_epochs):
    running_loss = 0.0

    for ppg_segment, ecg_segment in train_loader:
        ppg_segment = ppg_segment.unsqueeze(1)
        ecg_segment = ecg_segment.unsqueeze(1)

        output = model(ppg_segment)
        output = output.unsqueeze(1)
        loss = criterion(output, ecg_segment)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    loss_set.append(running_loss)
    print(f'Epoch: {epoch+1},  Loss: {running_loss}')
print('Finished Training')

model.eval()

sim1_ECG_test = []
tester = [101, 102, 103]

ECG_simulated1 = []
for ppg_segment, ecg_segment in dataset:
    ppg_segment = torch.tensor(ppg_segment).unsqueeze(0).unsqueeze(1)
    ecg_segment = torch.tensor(ecg_segment).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        output = model(ppg_segment).squeeze().numpy()
        ECG_simulated1.append(output)

con = 0
for ppg_segment, ecg_segment in test_dataset:
    if con < 0:
        ppg_segment = torch.tensor(ppg_segment).unsqueeze(0).unsqueeze(1)
        ecg_segment = ecg_segment

        with torch.no_grad():
            output = model(ppg_segment).squeeze().numpy()

        x1 = np.arange(len(ppg_segment.squeeze().numpy()))/fs
        x2 = np.arange(len(ecg_segment))/fs

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 1, 1)
        plt.plot(x1, ppg_segment.squeeze().numpy(),'k', label = 'Input PPG')
        plt.xlabel('Time (sec)', fontsize = 14)
        plt.title('PPG', fontsize = 20)
        plt.subplot(2, 1, 2)
        plt.plot(x2, ecg_segment, label = 'Normalized ECG')
        plt.plot(x2, output, label = 'Simulated ECG')
        #plt.plot(x2, test_ecg_segment-np.min(test_ecg_segment), label='normalized Actual ECG')
        #plt.plot(x2, ECG_WFset[i], label='Actual ECG')
        #plt.ylim(-0.45,1)
        plt.xlabel('Time (sec)', fontsize = 14)
        plt.title('ECG', fontsize = 20)
        plt.legend(fontsize='16')
        plt.show()
        # plt.savefig('test.jpg')
        con += 1


'''second CNN model'''
from CNNmodel_2 import nECGtoECGDataset, CNN2Model

dataset2 = nECGtoECGDataset(ECG_simulated1, ECG_WFset)
indices2 = list(range(len(dataset2)))
train_indices2 = indices2[:train_size]
test_indices2 = indices2[train_size:]
train_dataset2 = Subset(dataset2, train_indices2)
test_dataset2 = Subset(dataset2, test_indices2)
train_loader2 = DataLoader(dataset=train_dataset2, batch_size=32, shuffle=True)
test_loader2 = DataLoader(dataset=test_dataset2, batch_size=32, shuffle=False)

model2 = CNN2Model()
criterion2 = nn.MSELoss(reduction='sum')
optimizer2 = optim.Adam(model2.parameters(),lr=0.001)

num_epochs = 100
loss_set = []
for epoch in range(num_epochs):
    running_loss = 0.0

    for ecg_sim, ecg_act in train_loader2:
        ecg_sim = ecg_sim.unsqueeze(1)
        ecg_act = ecg_act.unsqueeze(1)
        output = model2(ecg_sim)
        output = output.unsqueeze(1)
        loss2 = criterion2(output, ecg_act)

        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()

        running_loss += loss2.item()

    loss_set.append(running_loss)
    print(f'Epoch: {epoch+1},  Loss: {running_loss}')
print('Finished Training')

model2.eval()
con = 0
for ecg_sim, ecg_act in test_dataset2:
    if con <0:
        ecg_sim = torch.tensor(ecg_sim).unsqueeze(0).unsqueeze(1)
        #ecg_act = torch.tensor(ecg_act).unsqueeze(0).unsqueeze(1)
        with torch.no_grad():
            output = model2(ecg_sim).squeeze().numpy()

        ecg_sim = np.array(ecg_sim.squeeze().numpy())
        ecg_sim = ecg_sim-np.mean(ecg_sim)
        x1 = np.arange(150)/fs
        x2 = np.arange(len(ecg_sim))/fs

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 1, 1)
        #plt.plot(x1, PPG_WFset[i], 'r', label='input PPG')
        #plt.xlabel('Time (sec)', fontsize = 14)
        #plt.title('PPG', fontsize = 20)
        plt.subplot(2, 1, 2)
        #plt.plot(x2, test_ecg_segment, label = 'Normalized ECG')
        plt.plot(x2, ecg_sim, label='Simulated 1 ECG')
        plt.plot(x2, output, label = 'Simulated 2 ECG')
        plt.plot(x2, ecg_act, label='Actual ECG')
        #plt.ylim(-0.45,1)
        plt.xlabel('Time (sec)', fontsize = 14)
        plt.title('ECG', fontsize = 20)
        plt.legend(fontsize='16')
        plt.show()
        con += 1


    # plt.savefig('test.jpg')

ECG_simulated2 = []
for ecg_sim, ecg_act in dataset2:
    ecg_sim = torch.tensor(ecg_sim).unsqueeze(0).unsqueeze(1)
    ecg_act = torch.tensor(ecg_act).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        output = model2(ecg_sim).squeeze().numpy()
        ECG_simulated2.append(output)

if 0:
    for i in ll:
        plt.plot(ECG_WFset[i], label='actual ECG',linewidth=3.0)
        plt.plot(ECG_simulated2[i], label='simulated2 ECG', linewidth=2.0)
        plt.plot(ECG_simulated1[i], label='simulated1 ECG', linewidth=1.0)
        plt.show()


sim_ecg_10s = []
for i in range(0,len(ECG_segments)):
    ecg_temp = np.array(ECG_segments[i])
    x = np.arange(len(ecg_temp))/100
    ll = [index for index, value in enumerate(segm_id) if value == i]
    peaks_input = [ECG_simulated2[i] for i in ll]
    p = [peaks_set[u] for u in ll]
    if i <len(ECG_segments) and i>len(ECG_segments)*0.9:
        pic = 0
    else:
        pic = 0
    new_ecg = recombine_peaks(p, ecg_temp, peaks_input, pic)
    sim_ecg_10s.append(new_ecg)

'''third CNN model'''
from CNNmodel_3 import ECGsegmentDataset, CNN3Model

ECG_segments_2 = [element[100:-100] for element in ECG_segments]
sim_ecg_10s_2 = [element[100:-100] for element in sim_ecg_10s]
dataset3 = ECGsegmentDataset(sim_ecg_10s_2, ECG_segments_2)
train_size = int(0.8*len(dataset3))
test_size = len(dataset3) - train_size
indices3 = list(range(len(dataset3)))
train_indices3 = indices3[:train_size]
test_indices3 = indices3[train_size:]
train_dataset3 = Subset(dataset3, train_indices3)
test_dataset3 = Subset(dataset3, test_indices3)
train_loader3 = DataLoader(dataset=train_dataset3, batch_size=32, shuffle=True)
test_loader3 = DataLoader(dataset=test_dataset3, batch_size=32, shuffle=False)


model3 = CNN3Model()
criterion3 = nn.MSELoss(reduction='sum')
optimizer3 = optim.Adam(model3.parameters(),lr=0.005)

num_epochs = 30
loss_set = []
for epoch in range(num_epochs):
    running_loss = 0.0

    for ecg_sim, ecg_act in train_loader3:
        ecg_sim = ecg_sim.unsqueeze(1)
        ecg_act = ecg_act.unsqueeze(1)
        output = model3(ecg_sim)
        output = output.unsqueeze(1)
        loss3 = criterion3(output, ecg_act)

        loss3.backward()
        optimizer3.step()
        optimizer3.zero_grad()

        running_loss += loss3.item()

    loss_set.append(running_loss)
    print(f'Epoch: {epoch+1},  Loss: {running_loss}')
print('Finished Training')

model3.eval()

from pre_process import butter_bandpass_filter
con = 0
for ecg_sim, ecg_act in test_dataset3:
    ecg_sim2 = torch.tensor(ecg_sim).unsqueeze(0).unsqueeze(1)
    ecg_act = ecg_act
    with torch.no_grad():
        output = model3(ecg_sim2).squeeze().numpy()

    output_f = butter_bandpass_filter(output, 0.1, 20, 100, order=3)
    x = np.arange(len(output))/fs
    if con < 10:
        #plt.figure(figsize=(15, 15))
        #plt.subplot(2, 1, 1)
        #plt.plot(x, PPG_segments[i], 'k', label='input PPG')
        #plt.xlabel('Time (sec)', fontsize = 14)
        #plt.title('PPG', fontsize = 20)
        #plt.subplot(2, 1, 2)
        plt.plot(x, ecg_act, 'k', label='Actual ECG', linewidth=4.0)
        plt.plot(x, output, 'b--', label = 'Simulated 2 ECG', linewidth=3.0)
        plt.plot(x, output_f, 'r', label='filtered Simulated 2 ECG', linewidth=3.0)
        plt.plot(x, ecg_sim, 'g--', label='Simulated 1 ECG', linewidth=1.5)
        plt.xlabel('Time (sec)', fontsize = 14)
        plt.title('ECG', fontsize = 20)
        plt.legend(fontsize='16')
        plt.show()
        con += 1







'''
class simECGhDataset(Dataset):
    def __init__(self, PPG_WFset, hp, he):
        self.PPG_WFset = [torch.tensor(ppg, dtype=torch.float32) for ppg in PPG_nWFset]
        self.hp = torch.tensor(hp, dtype=torch.float32)
        self.he = torch.tensor(he, dtype=torch.float32)

    def __len__(self):
        return len(self.he)

    def __getitem__(self, idx):
        PPG_WFset = self.PPG_WFset[idx]
        hp = self.hp[idx]
        he = self.he[idx]
        features = torch.cat((PPG_WFset, torch.tensor([hp])), dim=0)
        return features, he

dataset2 = simECGhDataset(PPG_WFset, hp_set, he_set)
dataloader2 = DataLoader(dataset2, batch_size=30, shuffle=True)

class simECGHModel(nn.Module):
    def __init__(self):
        super(simECGHModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(151, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model2 = simECGHModel()
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epochs = 10
loss_set = []
for epoch in range(num_epochs):
    running_loss = 0.0

    for inputs, targets in dataloader2:
        optimizer.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_set.append(running_loss)
    print(f'Epoch: {epoch+1},  Loss: {running_loss}')
print('Finished Training')

model2.eval()
for i in [100, 101, 102]:
    with torch.no_grad():
        n_ecg = torch.tensor(ECG_simulated1[i], dtype=torch.float32)
        PPG_input = torch.tensor(PPG_WFset[i], dtype=torch.float32)
        hp = torch.tensor([hp_set[i]], dtype=torch.float32)
        features = torch.cat((PPG_input, hp), dim=0)
        sim_ECGH = model2(features)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 1, 1)
    plt.plot(x1, PPG_WFset[i], 'r', label='normalized PPG')
    plt.xlabel('Time (sec)', fontsize = 14)
    plt.title('PPG', fontsize = 20)
    plt.subplot(2, 1, 2)
    #plt.plot(x2, test_ecg_segment, label = 'Normalized ECG')
    plt.plot(x2, ECG_simulated1[i], label = 'Simulated ECG 1')
    plt.plot(x2, ECG_simulated1[i]*sim_ECGH, label='Simulated ECG * simulated Height')
    plt.plot(x2, ECG_WFset[i]-np.min(ECG_WFset[i]), label='Actual ECG')
    #plt.ylim(-0.45,1)
    plt.xlabel('Time (sec)', fontsize = 14)
    plt.title('ECG', fontsize = 20)
    plt.legend(fontsize='16')
    plt.show()
    '''