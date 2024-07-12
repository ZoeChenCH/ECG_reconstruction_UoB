import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
class nECGtoECGDataset(Dataset):
    def __init__(self,ECG_simulated1, ECG_segments):
        self.ECG_simulated1 = ECG_simulated1
        self.ECG_segments = ECG_segments

    def __len__(self):
        return len(self.ECG_simulated1)

    def __getitem__(self, idx):
        ECG_sim = np.array(self.ECG_simulated1[idx]).astype(np.float32)
        ECG_segment = np.array(self.ECG_segments[idx]).astype(np.float32)
        return ECG_sim, ECG_segment

class CNN2Model(nn.Module):
    def __init__(self):
        super(CNN2Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

        self._initialize_fc()

        self.fc1 = nn.Linear(self.fc_input_dim, 52)
        self.fc2 = nn.Linear(52, 100)


    def _initialize_fc(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 100)
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
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x