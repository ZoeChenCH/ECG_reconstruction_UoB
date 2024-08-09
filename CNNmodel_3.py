import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
class ECGsegmentDataset(Dataset):
    def __init__(self,sim_ecg_10s, ECG_segments):
        self.sim_ecg_10s = sim_ecg_10s
        self.ECG_segments = ECG_segments
    def __len__(self):
        return len(self.sim_ecg_10s)

    def __getitem__(self, idx):
        sim_ecg_10s = np.array(self.sim_ecg_10s[idx]).astype(np.float32)
        ECG_segments = np.array(self.ECG_segments[idx]).astype(np.float32)
        return sim_ecg_10s, ECG_segments

class CNN3Model(nn.Module):
    def __init__(self):
        super(CNN3Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self._initialize_fc()

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 500)


    def _initialize_fc(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 500)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            self.fc_input_dim = x.numel()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x