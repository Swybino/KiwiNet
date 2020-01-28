import argparse
import pickle
from datetime import date
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import config
import utils.utils as utils
from dataset import FoADataset, ToTensor, Normalization
from kiwiNet import Kiwi
from utils.confusion_matrix import ConfusionMatrix
from utils.video import Video
from utils.viewer import Viewer
import os


def accuracy(net, test_loader, *, confusion_matrix=True, visualize=False):
    correct = 0
    total = 0
    cm = ConfusionMatrix()
    model.eval()
    with torch.no_grad():

        for i, test_data in enumerate(test_loader):
            inputs, labels = test_data['inputs'], test_data['labels']
            # print(inputs, labels)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # display_results(test_data, predicted)
            names_outputs = output_processing(outputs, test_data['names_list'])

            if confusion_matrix:
                cm.add_results(test_data['name_label'], names_outputs)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    cm.normalize()
    print("Accuracy: %0.4f" % (correct / total), total, correct, cm,  sep="\n")
    return correct / total, cm


def save_results(data, frame):
    with open("", 'wb') as f:
        pickle.dump(data, f)
    print("History file written")
    return


def display_results(data, predicted):
    print(predicted)
    for i in range(predicted.size(0)):
        video_path = os.path.join(config.video_root, "%s.MP4" %data["video"][i])
        v = Viewer(Video(video_path)[data["frame"][i]])
        print("######", data["name_label"][i])
        kid, focus = data["positions"][i][0:2], data["positions"][i][2*predicted[i]:2+2*predicted[i]]
        v.plt_results(kid, focus)
        v.show()
    return

def output_processing(outputs, names_list):
    r = []
    _, predicted = torch.max(outputs.data, 1)
    for j in range(predicted.size(0)):
        if predicted[j] == 0:
            r.append('z')
        else:
            r.append(names_list[predicted[j]][j])
    return r


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

    args = parser.parse_args()
    suffix = utils.build_suffix(args.structure)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.train_set is not None:
        train_set_file = args.train_set
    else:
        train_set_file = "data/labels/train_labels_frame_patches.csv"

    train_set = FoADataset(train_set_file, "data/inputs",
                           transform=transforms.Compose([
                               Normalization(),
                               ToTensor()
                           ]))

    if args.test_set is not None:
        test_set_file = args.test_set
    else:
        test_set_file = "data/labels/test_labels_frame_patches.csv"

    test_set = FoADataset(test_set_file, "data/inputs",
                           transform=transforms.Compose([
                               Normalization(),
                               ToTensor()
                           ]))

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True, num_workers=4)

    model = Kiwi(config.nb_kids, args.structure)
    model.cuda()
    if args.load_state is not None:
        model.load_state_dict(torch.load(args.load_state))
    today = date.today()
    model_save_path = "model/model%s_%s.pt" % (suffix, today)
    history_save_path = "model/history%s_%s.p" % (suffix, today)

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.25, 1, 1, 1, 1, 1]))
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss_history = []

    # accuracy(model, test_loader)
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        model.train(True)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, eye_img, target = data['inputs'], data['eye_img'], data['labels']
            inputs.cuda()
            eye_img.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs, eye_img)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % args.print_rate == args.print_rate-1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.print_rate))
                running_loss = 0.0

            loss_history.append((epoch, i, round(loss.item(), 3)))
        torch.save(model.state_dict(), model_save_path)
        save_history(history_save_path, loss_history)
        if epoch % args.accuracy_rate == args.accuracy_rate-1:
            accuracy(model, test_loader)

    print('Finished Training')
