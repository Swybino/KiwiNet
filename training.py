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


def accuracy(net, test_loader):
    correct = 0
    total = 0
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for test_data in test_loader:
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
        argmax = output[idx].max(0)[1]
        if argmax == idx:
            result[name] = 'z'
        else:
            result[name] = names_list[argmax]


def save_history(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("History file written")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiwi Training')
    parser.add_argument('-s', '--structure', nargs='+', type=int, help='Structure of the model', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--state_dict', type=str, help="state directory file to load before training")

    # parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')

    args = parser.parse_args()
    print(args.structure)
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
    train_loader = DataLoader(train_set, batch_size=16,
                              shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=8,
                             shuffle=False, num_workers=4)

    model = Kiwi(config.nb_kids, args.structure)
    if args.state_dict is not None:
        model.load_state_dict(torch.load(args.state_dict))

    model_save_path = "model/model%s.pt" % suffix
    history_save_path = "model/history%s.p" % suffix

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
            loss = multi_entropy_loss(outputs, target)
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




