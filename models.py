import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(42, 32)
        self.hidden1 = nn.Linear(32, 16)
        self.hidden2 = nn.Linear(16,1)
        self.dropout = nn.Dropout(p=.2)
    def forward(self, x):
        a1 = F.relu(self.first_layer(x))
        a1 = self.dropout(a1)
        a2 = F.relu(self.hidden1(a1))
        a2 = self.dropout(a2)
        return F.relu(self.hidden2(a2))

# Basic CNN
class BaseCNN(nn.Module):
    def __init__(self, in_channels):
        super(BaseCNN, self).__init__()
        self.in_channels = in_channels

        self.c1 = nn.Conv1d(in_channels, 1,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(1)
                   
        self.drop = nn.Dropout(p=.25)
        self.pred1 = nn.Linear(14, 16)
        self.final = nn.Linear(16, 1)
    def forward(self, x):
        a_c1 = self.drop(self.bn1(self.c1(x)))
        
        a1 = F.relu(self.pred1(a_c1))
        return F.relu(self.final(a1))

# Advanced CNN
class AdvCNN(nn.Module):
    def __init__(self, in_channels):
        super(AdvCNN, self).__init__()
        self.in_channels = in_channels

        self.c1 = nn.Conv1d(in_channels, in_channels*2,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels*2)
        self.pool = nn.MaxPool1d(2)
        self.c2 = nn.Conv1d(in_channels*2, in_channels*2*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels*2*2)
                   
        self.drop = nn.Dropout(p=.25)
        self.pred1 = nn.Linear(36, 16)
        self.final = nn.Linear(16, 1)
    def forward(self, x):
        a_c1 = self.drop(self.pool(self.bn1(self.c1(x))))
        a_c2 = self.drop(self.pool(self.bn2(self.c2(a_c1))))
        flat = a_c2.view((a_c2.size(0), a_c2.size(1)*a_c2.size(2)))
        a1 = F.relu(self.pred1(flat))
        return F.relu(self.final(a1))