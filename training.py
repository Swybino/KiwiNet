from dataset import FoADataset, RandomPermutations, Binarization, ToTensor, Normalization
from kiwiNet import Kiwi
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import config
from torch.utils.data import Dataset, DataLoader


def custom_criterion(output, target):
    criterion = nn.CrossEntropyLoss()
    losses = []
    target = target.type(torch.LongTensor)
    for i in range(6):
        print(output[:, i, :], target[:, i, :])
        losses.append(criterion(output[:, i, :], target[:, i, :]))
    print(losses)
    return losses


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # print(device)

    dataset = FoADataset("data/labels/171214_1.csv", "data/171214_1/correction_angle", "171214_1",
                         transform=transforms.Compose([
                             RandomPermutations(),
                             Binarization(size=6),
                             Normalization(),
                             ToTensor()
                         ]))

    train_length = int(len(dataset) * 0.8)
    lengths = [train_length, len(dataset) - train_length]
    train_set, test_set = torch.utils.data.random_split(dataset, lengths)

    train_loader = DataLoader(train_set, batch_size=2,
                              shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=2,
                             shuffle=False, num_workers=8)

    net = Kiwi(config.nb_kids)
    # criterion = nn.MultiLabelMarginLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, target = data['inputs'], data['out']
            # zero the parameter gradientse
            # optimizer.zero_grad()

            outputs = net(inputs)
            # outputs = outputs.view(-1, 36)
            # labels = labels.view(-1, 36)
            print(outputs, outputs.type(), target, target.type(), sep="\n")
            custom_criterion(outputs, target)
            target = target.type(torch.LongTensor)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        torch.save(net.state_dict(), "model/model.pt")

    print('Finished Training')
