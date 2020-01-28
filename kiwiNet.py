import torch
import torch.nn as nn
import torchvision.models as models

batch_size = 10


class Kiwi(nn.Module):
    def __init__(self, kids_nb, param_list):
        super(Kiwi, self).__init__()
        self.kids_nb = kids_nb
        self.in_shape = self.kids_nb * 2 + 3
        n_in, n_out = self.kids_nb * 2 + 3, self.kids_nb
        param_list = param_list
        self.layers_list = []
        for i in range(len(param_list) - 1):
            self.layers_list.append(nn.Linear(param_list[i], param_list[i + 1]))
            self.layers_list.append(nn.ReLU())
            self.layers_list.append(nn.BatchNorm1d(param_list[i + 1]))

        vgg16 = models.vgg16(pretrained=True)
        self.eye_features = nn.Sequential(vgg16.features, vgg16.avgpool)
        self.fc = nn.Sequential(nn.Linear(25088, param_list[0] - n_in), nn.ReLU(), nn.BatchNorm1d(param_list[0] - n_in))
        self.core = nn.Sequential(*self.layers_list)
        self.out1 = nn.Linear(param_list[-1], n_out)

    def forward(self, x, e):
        x = x.type(torch.FloatTensor)
        x = x.view(-1, self.in_shape)
        e = self.eye_features(e)
        e = e.view(-1, 25088)
        e = self.fc(e)
        x = torch.cat([e.cuda(), x.cuda()], dim=1)
        x = self.core(x)
        x = self.out1(x)
        x = x.view(-1, self.kids_nb)
        return x


if __name__ == '__main__':
    kiwi = Kiwi(6, [512, 512])
    print(kiwi)
