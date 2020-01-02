from dataset import FoADataset, RandomPermutations, Binarization, ToTensor, Normalization
from kiwiNet import Kiwi
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import config
from torch.utils.data import Dataset, DataLoader


def custom_criterion(output, target):
    # nn.functional.binary_cross_entropy(input, target)
    # criterion = nn.CrossEntropyLoss()
    # target = target.type(torch.LongTensor)
    losses = []
    for i in range(6):
        print(output[:, i, :], target[:, i, :])
        losses.append(nn.functional.binary_cross_entropy(output[:, i, :], target[:, i, :]))
    print(losses)
    losses = torch.Tensor(losses)
    loss = torch.mean(losses)
    print(loss)
    print(nn.functional.binary_cross_entropy(output, target))
    return nn.functional.binary_cross_entropy(output, target)


def output_processing(output, names_list):
    result = {}
    for idx, name in enumerate(names_list):
        argmax = output[idx].max(0)[1]
        if argmax == idx:
            result[name] = 'z'
        else:
            result[name] = names_list[argmax]


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # print(device)

    dataset = FoADataset("data/labels/171214_1.csv", "data/171214_1/correction_angle_100_50", "171214_1",
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
                              shuffle=False, num_workers=4)

    net = Kiwi(config.nb_kids)
    # criterion = nn.MultiLabelMarginLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)

    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, target = data['inputs'], data['out']
            # zero the parameter gradientse
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = nn.functional.binary_cross_entropy(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        torch.save(net.state_dict(), "model/model.pt")

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
