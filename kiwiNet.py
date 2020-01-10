import torch
import torch.nn as nn
import torch.nn.functional as F
import config

batch_size = 10

class Kiwi(nn.Module):
    def __init__(self, kids_nb, param_list):
        super(Kiwi, self).__init__()
        self.kids_nb = kids_nb
        n_in, n_out = self.kids_nb * 7, self.kids_nb * (self.kids_nb+1)
        param_list = [n_in] + param_list + [n_out]
        self.layers_list = []
        for i in range(len(param_list)-1):
            self.layers_list.append(nn.Linear(param_list[i], param_list[i+1]))
            self.layers_list.append(nn.ReLU())
            self.layers_list.append(nn.BatchNorm1d(param_list[i+1]))
        self.layers_list.pop()
        self.layers_list.pop()
        self.core = nn.Sequential(*self.layers_list)

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x = x.view(-1, self.kids_nb * 7)
        x = self.core(x)
        x = x.view(-1, self.kids_nb, self.kids_nb+1)
        return x
