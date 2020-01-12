import argparse
import pickle

import utils
from dataset import FoADataset, RandomPermutations, Binarization, ToTensor, Normalization
from kiwiNet import Kiwi
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import config
from torch.utils.data import Dataset, DataLoader


class MultiEntropyLoss:
    def __init__(self, size=6, weight=0.2):
        self.bce = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.ce_list = []
        for i in range(size):
            w = torch.ones(size)
            w[i] = weight
            self.ce_list.append(nn.CrossEntropyLoss(weight=w))

    def get_loss(self, output, target):
        losses = []
        for i in range(output.size(1)):
            losses.append(self.ce_list[i](output[:, i, 1:], target[:, i, 1]))
            losses.append(self.bce(self.sigmoid(output[:, i, 0]), target[:, i, 0].type(torch.FloatTensor)))

        # return loss
        return sum(losses)

def multi_entropy_loss(output, target):
    """
    Compute the mean cross entropy of the
    :param output: output of the network
    :type output: torch.Tensor
    :param target: target
    :type target: torch.Tensor
    :return:
    :rtype:
    """
    bce = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    losses = []
    for i in range(output.size(1)):
        w = torch.ones(output.size(1))
        w[i] = 0.20
        ce = nn.CrossEntropyLoss(weight=w)
        losses.append(ce(output[:, i, 1:], target[:, i, 1]))
        losses.append(bce(sigmoid(output[:, i, 0]), target[:, i, 0].type(torch.FloatTensor)))

    # return loss
    return sum(losses)


def sum_cross_entropy_loss(output, target):
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

    loss = sum(losses)
    # loss = torch.mean(torch.stack(losses))
    return loss


def accuracy(net, test_loader, visualize=False):
    correct = 0
    total = 0
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            inputs, target = test_data['inputs'], test_data['labels']
            outputs = net(inputs)

            look_bool = sigmoid(outputs.data[:, :, 0]) > 0.5
            _, predicted = torch.max(outputs.data[:, :, 1:], 2)

            total += target.size(0) * target.size(1)
            correct += ((look_bool == target[:, :, 0]) & ((target[:, :, 0] == 0) | (predicted == target[:, :, 1]))).sum().item()
    return correct / total


def output_processing(output, names_list):
    result = {}
    for idx, name in enumerate(names_list):
        if output[idx,0] < 0:
            result[name] = 'z'
        else:
            argmax = output[idx,1:].max(0)[1]
            result[name] = 'z' if argmax == idx else names_list[argmax]
    return result


def save_history(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("History file written")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiwi Training')
    parser.add_argument('-s', '--structure', nargs='+', type=int, help='Structure of the model', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--state_dict', type=str, help="state directory file to load before training")

    args = parser.parse_args()
    suffix = utils.build_suffix(args.structure)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    for i, data in train_loader:
        print()
    for i, data in test_loader:
        print()

    model = Kiwi(config.nb_kids, args.structure)
    if args.state_dict is not None:
        model.load_state_dict(torch.load(args.state_dict))

    model_save_path = "model/model%s.pt" % suffix
    history_save_path = "model/history%s.p" % suffix

    criterion = MultiEntropyLoss(config.nb_kids)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_history = []

    print("Accuracy: %0.4f" % accuracy(model, test_loader))
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, target = data['inputs'], data['labels']
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion.get_loss(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

            loss_history.append((epoch, i, round(loss.item(), 3)))
        torch.save(model.state_dict(), model_save_path)
        save_history(history_save_path, loss_history)
        if epoch % 10 == 9:
            print("Accuracy: %0.4f" % accuracy(model, test_loader))

    print('Finished Training')




