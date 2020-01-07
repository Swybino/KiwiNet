import argparse
import pickle

from dataset import FoADataset, RandomPermutations, Binarization, ToTensor, Normalization
from kiwiNet import Kiwi
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import config
from torch.utils.data import Dataset, DataLoader


def mean_cross_entropy_loss(output, target):
    """
    Compute the mean cross entropy of the
    :param output: output of the network
    :type output: torch.Tensor
    :param target: target
    :type target: torch.Tensor
    :return:
    :rtype:
    """
    criterion = nn.CrossEntropyLoss()
    losses = []
    for i in range(output.size(1)):
        # print(output[:, i, :], target[:, i])
        losses.append(criterion(output[:, i, :], target[:, i]))

    loss = torch.mean(torch.stack(losses))
    return loss


def accuracy(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['inputs'], data['out']
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 2)
            total += target.size(0) * target.size(1)
            correct += (predicted == labels).sum().item()
    return correct / total


def output_processing(output, names_list):
    result = {}
    for idx, name in enumerate(names_list):
        argmax = output[idx].max(0)[1]
        if argmax == idx:
            result[name] = 'z'
        else:
            result[name] = names_list[argmax]


def save_history(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("file written")


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # print(device)

    dataset = FoADataset("data/labels", "data/inputs",
                         transform=transforms.Compose([
                             RandomPermutations(),
                             Binarization(size=6),
                             Normalization(),
                             ToTensor()
                         ]))

    train_length = int(len(dataset) * 0.8)
    lengths = [train_length, len(dataset) - train_length]
    train_set, test_set = torch.utils.data.random_split(dataset, lengths)
    test_loader = DataLoader(test_set, batch_size=4,
                             shuffle=False, num_workers=1)

    train_loader = DataLoader(train_set, batch_size=16,
                              shuffle=False, num_workers=8)

    net = Kiwi(config.nb_kids, [512, 512,1024,1024,1024])
    # net.load_state_dict(torch.load("model/model.pt"))

    # criterion = nn.MultiLabelMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_history = []
    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, target = data['inputs'], data['out']
            # zero the parameter gradientse
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = mean_cross_entropy_loss(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

            loss_history.append((epoch, i, round(loss.item(), 3)))
        torch.save(net.state_dict(), "model/model.512x2.1024x3.pt")

        save_history("model/history.512x2.1024x3.pickle", loss_history)
        if epoch % 10 == 9:
            print("Accuracy: %0.4f" % accuracy(net, test_loader))

    print('Finished Training')


    parser = argparse.ArgumentParser(description='Kiwi Training')

    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')

    args = parser.parse_args()
