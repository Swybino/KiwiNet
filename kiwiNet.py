import torch
import torch.nn as nn

batch_size = 10


class Kiwi(nn.Module):
    def __init__(self, kids_nb, param_list):
        super(Kiwi, self).__init__()
        self.kids_nb = kids_nb
        self.in_shape = self.kids_nb * 2 + 3
        n_in, n_out = self.kids_nb * 2 + 3, self.kids_nb
        param_list = [n_in] + param_list
        self.layers_list = []
        for i in range(len(param_list)-1):
            self.layers_list.append(nn.Linear(param_list[i], param_list[i+1]))
            self.layers_list.append(nn.ReLU())
            self.layers_list.append(nn.BatchNorm1d(param_list[i+1]))
        self.core = nn.Sequential(*self.layers_list)
        self.out1 = nn.Linear(param_list[-1], n_out)

    def forward(self, x):
        x = x.view(-1, self.in_shape)
        x = self.core(x)
        x = self.out1(x)
        x = x.view(-1, self.kids_nb)
        return x
