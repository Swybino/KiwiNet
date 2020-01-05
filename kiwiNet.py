import torch
import torch.nn as nn
import torch.nn.functional as F
import config

batch_size = 10


class Kiwi(nn.Module):
    def __init__(self, kids_nb):
        self.kids_nb = kids_nb
        n_in, n_out = self.kids_nb * 7, self.kids_nb ** 2
        n_1, n_2, n_3 = 256, 512, 1024
        super(Kiwi, self).__init__()
        self.fc1 = nn.Linear(n_in, n_1)
        self.bn1 = nn.BatchNorm1d(n_1)
        self.fc2 = nn.Linear(n_1, n_2)
        self.bn2 = nn.BatchNorm1d(n_2)
        self.fc3 = nn.Linear(n_2, n_3)
        self.bn3 = nn.BatchNorm1d(n_3)
        self.fc4 = nn.Linear(n_3, n_out)

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x = x.view(-1, self.kids_nb * 7)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        # x = self.bn4(x)
        x = x.view(-1, self.kids_nb, self.kids_nb)

        return x
