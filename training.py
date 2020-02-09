import argparse
import os
import pickle
import timeit
from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import config
import utils.utils as utils
from dataset import FoADataset, ToTensor, RandomTranslation, RandomPermutations, RandomRotation
from kiwiNet import Kiwi
from utils.confusion_matrix import ConfusionMatrix


def accuracy(net, test_loader, *, confusion_matrix=True, save=False):
    correct = 0
    total = 0
    cm = ConfusionMatrix()
    model.eval()
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            inputs, labels = test_data['inputs'], test_data['labels']
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            names_outputs = utils.output_processing(outputs, test_data['names_list'])
            if save:
                utils.save_results(results_save_path, test_data['video'], test_data['frame'],
                                   test_data['name'], names_outputs)

            if confusion_matrix:
                cm.add_results(test_data['name_label'], names_outputs)
            # print(test_data['name_label'], type(test_data['name_label']))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    cm.normalize()

    if save:
        msg = "Epochs: %d\nAccuracy: %0.4f\nTotal: %d\nCorrect:%d\nConfusion matrix:\n%s" % (args.epochs, correct/total, total, correct, cm)
        with open(accuracy_save_path, 'w') as f:
            f.write(str(msg))
    print("Accuracy: %0.4f" % (correct / total), total, correct, cm, sep="\n")

    return correct / total, cm


def save_history(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("History file written")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiwi Training')
    parser.add_argument('-s', '--structure', nargs='+', type=int, help='Structure of the model', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-l', '--load_state', type=str, help="state directory file to load before training")
    parser.add_argument('--train_set', type=str, help="train set location")
    parser.add_argument('--test_set', type=str, help="test set location")
    parser.add_argument('--print_rate', type=int, default=200, help='print every * mini epochs')
    parser.add_argument('--accuracy_rate', type=int, default=10, help='tests accuracy every * epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of data per batch')
    parser.add_argument('-t', '--test_save', action='store_true', help="testing save file")
    parser.add_argument('-w', '--weights', nargs='+', type=float, help="weight")

    args = parser.parse_args()
    suffix = utils.build_suffix(args.structure)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    if args.train_set is not None:
        train_set_file = args.train_set
    else:
        train_set_file = "data/labels/train_labels_frame_patches100.csv"

    train_set = FoADataset(train_set_file, config.inputs_dir,
                           transform=transforms.Compose([
                               RandomTranslation(),
                               RandomPermutations(),
                               ToTensor()
                           ]))

    if args.test_set is not None:
        test_set_file = args.test_set
    else:
        test_set_file = "data/labels/test_labels_frame_patches100.csv"

    test_set = FoADataset(test_set_file, config.inputs_dir,
                          transform=transforms.Compose([
                              ToTensor()
                          ]))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = Kiwi(config.nb_kids, args.structure)
    model.cuda()
    if args.load_state is not None:
        model.load_state_dict(torch.load(args.load_state))
    today = date.today()

    model_folder = "data/models"
    history_folder = "data/history"
    results_folder = "data/results"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(history_folder):
        os.makedirs(history_folder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    model_save_path = os.path.join(model_folder, "%s_model_%s.pt" % (today, suffix))
    history_save_path = os.path.join(history_folder, "%s_history_%s.p" % (today, suffix))
    results_save_path = os.path.join(results_folder, "%s_results_%s.csv" % (today, suffix))
    accuracy_save_path = os.path.join(results_folder, "%s_accuracy_%s.txt" % (today, suffix))

    if args.weights is not None and len(args.weights) == config.nb_kids:
        weights = args.weights
    else:
        weights = [0.3, 1.0, 1.0, 1.0, 1.0, 1.0]

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).cuda())
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss_history = []

    # accuracy(model, test_loader)
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        start = timeit.default_timer()
        model.train(True)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, target = data['inputs'], data['labels']
            inputs, target = inputs.cuda(), target.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % args.print_rate == args.print_rate - 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.print_rate))
                running_loss = 0.0

            loss_history.append((epoch, i, round(loss.item(), 3)))

        stop = timeit.default_timer()
        time = stop - start
        print('Epoch Time: %d:%d:%d' % (time // 360, (time % 3600) // 60, (time % 3600) % 60))

        torch.save(model.state_dict(), model_save_path)
        save_history(history_save_path, loss_history)
        if args.accuracy_rate > 0 and epoch % args.accuracy_rate == args.accuracy_rate - 1:
            accuracy(model, test_loader)

    accuracy(model, test_loader, save=args.test_save)
    print('Finished Training')
