import numpy as np
import matplotlib.pylab as plt

row_file1 = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case01/fulldata/uq_vsd_case01_fulldata_02.csv')
row_file = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case02/uq_vsd_case02_fulldata_01.csv')
row_file2 = ('/Users/zhao-weichen/Zoe/ECG reconstruction/uqvitalsignsdata/case01/fulldata/uq_vsd_case01_fulldata_03.csv')
fs = 100
dur = 10 * fs
wl = fs*1.5

from pre_process import getData, create_segments, recombine_peaks
ECG1, PPG1 = getData(row_file)
ECG2, PPG2 = getData(row_file1)
ECG_test, PPG_test = getData(row_file1)

ECG_segments1, PPG_segments1, PPG_WFset1, ECG_WFset1, peaks_set1, PPG_nWFset1, ECG_nWFset1, hp_set1, he_set1= create_segments(PPG1, ECG1, dur, wl)
ECG_segments2, PPG_segments2, PPG_WFset2, ECG_WFset2, peaks_set2, PPG_nWFset2, ECG_nWFset2, hp_set2, he_set2= create_segments(PPG2, ECG2, dur, wl)
ECG_segmentsT, PPG_segmentsT, PPG_WFsetT, ECG_WFsetT, peaks_setT, PPG_nWFsetT, ECG_nWFsetT, hp_setT, he_setT= create_segments(PPG_test, ECG_test, dur, wl)

PPG_WFset = PPG_WFset2
ECG_WFset = ECG_WFset2
peaks_set = peaks_set2
PPG_nWFset = PPG_nWFset2
ECG_nWFset = ECG_nWFset2
hp_set = hp_set2
he_set = he_set2

ppg_temp = np.array(PPG_segments1[0])
x = np.arange(len(ppg_temp))/100
p = peaks_set1[0]
new_ppg = recombine_peaks(p, ppg_temp, PPG_WFset1[0])

from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from CNNmodel import CNNModel,PPGtoECGDataset
from itertools import chain

#dataset = PPGtoECGDataset(PPG_segments, ECG_segments)
PPG_nWFset = list(chain.from_iterable(PPG_nWFset))
ECG_nWFset = list(chain.from_iterable(ECG_nWFset))
dataset = PPGtoECGDataset(PPG_nWFset, ECG_nWFset)
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
dataloader = DataLoader(dataset = train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(dataset = test_dataset, batch_size=50, shuffle=False)

model = CNNModel()
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(),lr=0.005)

num_epochs = 25
loss_set = []
for epoch in range(num_epochs):
    running_loss = 0.0

    for ppg_segment, ecg_segment in dataloader:
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

from plot_model1 import plot_model1
'''
for ppg, ecg in test_loader:
    ppg_segment = ppg.unsqueeze(1)
    ecg_segment = ecg.unsqueeze(1)
    with torch.no_grad():
        simulated_ecg_segment = model(ppg_segment).squeeze().numpy()
    x1 = np.arange(len(ppg_segment[0,0,:]))/fs
    x2 = np.arange(len(ecg_segment[0,0,:]))/fs
    ppg = ppg_segment.squeeze().numpy()
    ecg = ecg_segment.squeeze().numpy()
    for i in range(len(ppg)):
        plot_model1(x1, ppg[i,:], x2, simulated_ecg_segment[i,:], ecg[i,:])
'''

sim1_ECG_test = []
tester = [101, 102, 103]

for i in tester:
    test_ppg_segment = PPG_nWFset[i]
    test_ecg_segment = ECG_nWFset[i]
    test_ppg_segment = torch.tensor(test_ppg_segment).unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        simulated_ecg_segment = model(test_ppg_segment).squeeze().numpy()
        sim1_ECG_test.append(simulated_ecg_segment)
    x1 = np.arange(len(test_ppg_segment.squeeze().numpy()))/fs
    x2 = np.arange(len(test_ecg_segment))/fs
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 1, 1)
    plt.plot(x1, test_ppg_segment.squeeze(),'k', label = 'Input PPG')
    #plt.plot(x1, PPG_nWFset[i], 'r', label='normalized PPG')
    plt.xlabel('Time (sec)', fontsize = 14)
    plt.title('PPG', fontsize = 20)
    plt.subplot(2, 1, 2)
    plt.plot(x2, test_ecg_segment, label = 'Normalized ECG')
    plt.plot(x2, simulated_ecg_segment, label = 'Simulated ECG')
    #plt.plot(x2, test_ecg_segment-np.min(test_ecg_segment), label='normalized Actual ECG')
    plt.plot(x2, ECG_WFset[i], label='Actual ECG')
    #plt.ylim(-0.45,1)
    plt.xlabel('Time (sec)', fontsize = 14)
    plt.title('ECG', fontsize = 20)
    plt.legend(fontsize='16')
    plt.show()

    # plt.savefig('test.jpg')

ECG_simulated1 = []
for i in range(0,len(PPG_nWFset)):
    test_ppg_segment = PPG_nWFset[i]
    test_ecg_segment = ECG_nWFset[i]
    test_ppg_segment = torch.tensor(test_ppg_segment).unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        simulated_ecg_segment = model(test_ppg_segment).squeeze().numpy()
        ECG_simulated1.append(simulated_ecg_segment)



'''second CNN model'''
from CNNmodel_2 import nECGtoECGDataset, CNN2Model

dataset2 = nECGtoECGDataset(ECG_simulated1, ECG_WFset)
dataloader2 = DataLoader(dataset2, batch_size=50, shuffle=True)

model2 = CNN2Model()
criterion2 = nn.MSELoss(reduction='sum')
optimizer2 = optim.Adam(model2.parameters(),lr=0.01)

num_epochs = 25
loss_set = []
for epoch in range(num_epochs):
    running_loss = 0.0

    for ecg_sim, ecg_segment in dataloader2:
        ecg_sim = ecg_sim.unsqueeze(1)
        ecg_segment = ecg_segment.unsqueeze(1)

        output = model2(ecg_sim)
        output = output.unsqueeze(1)
        loss2 = criterion2(output, ecg_segment)

        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()

        running_loss += loss2.item()

    loss_set.append(running_loss)
    print(f'Epoch: {epoch+1},  Loss: {running_loss}')
print('Finished Training')

model2.eval()
con = 0
for i in tester:
    test_ecg_sim = sim1_ECG_test[con]
    test_ecg_segment = ECG_WFsetT[i]
    test_ecg_sim = torch.tensor(test_ecg_sim).unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        simulated_ecg_segment = model2(test_ecg_sim).squeeze().numpy()

    x1 = np.arange(150)/fs
    x2 = np.arange(len(test_ecg_segment))/fs
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 1, 1)
    plt.plot(x1, PPG_WFset[i], 'r', label='input PPG')
    plt.xlabel('Time (sec)', fontsize = 14)
    plt.title('PPG', fontsize = 20)
    plt.subplot(2, 1, 2)
    #plt.plot(x2, test_ecg_segment, label = 'Normalized ECG')
    plt.plot(x2, sim1_ECG_test[con], label='Simulated 1 ECG')
    plt.plot(x2, simulated_ecg_segment, label = 'Simulated 2 ECG')
    plt.plot(x2, test_ecg_segment, label='Actual ECG')
    #plt.ylim(-0.45,1)
    plt.xlabel('Time (sec)', fontsize = 14)
    plt.title('ECG', fontsize = 20)
    plt.legend(fontsize='16')
    plt.show()
    con += 1

    # plt.savefig('test.jpg')













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