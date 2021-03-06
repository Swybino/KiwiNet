import json
import os
from random import random
import utils.utils as utils
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config


class VideoDataset(Dataset):
    def __init__(self, root_dir, kids_count =6, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.kids_count= kids_count
        self.root_dir = root_dir
        self.transform = transform
        self.files_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.files_list * self.kids_count)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_idx = idx // self.kids_count
        kid_idx = idx % self.kids_count

        tmp = self.files_list[file_idx][:-5].split("_")
        frame = tmp[-1]
        video = "_".join(tmp[:-1])

        file_path = os.path.join(self.root_dir, self.files_list[file_idx])
        frame_data = utils.read_input(file_path)
        name = list(frame_data.keys())[kid_idx]
        name_list = []
        bboxes = []
        pose = [0, 0, 0]
        main_pos = [0, 0]
        for key, item in frame_data.items():
            if key == name:
                bbox = item[config.BBOX_KEY]
                main_pos = [bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]
                confidence = item[config.CONFIDENCE_KEY]
                pose = [x * confidence for x in item[config.POSE_KEY]]
                name_list.insert(0, key)
            else:
                bbox = item[config.BBOX_KEY]
                bboxes.append(bbox[0] + 0.5 * bbox[2])
                bboxes.append(bbox[1] + 0.5 * bbox[3])
                name_list.append(key)
        if len(bboxes) < 10:
            for i in range(10 - len(bboxes)):
                bboxes.append(0)
        if len(name_list) < 6:
            for i in range(6 - len(name_list)):
                name_list.append("z")
        inputs = np.array(pose + main_pos + bboxes)

        sample = {"inputs": inputs,
                  "frame": frame,
                  "name": name,
                  "names_list": name_list,
                  "video": video}

        if self.transform:
            sample = self.transform(sample)
        return sample


class FoADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        labels = self.labels.iloc[idx]

        # print(labels["video"],labels["frame"])
        file_path = os.path.join(self.root_dir, '%s_%s.json' % (labels["video"], labels["frame"]))

        with open(file_path, 'r') as f:
            frame_data = json.load(f)
        name_list = []
        bboxes = []
        pose = [0, 0, 0]
        main_pos = [0, 0]
        for key, item in frame_data.items():
            if key == labels["name"]:
                bbox = item[config.BBOX_KEY]
                main_pos = [bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]
                confidence = item[config.CONFIDENCE_KEY]
                pose = [x * confidence for x in item[config.POSE_KEY]]
                name_list.insert(0, key)
            else:
                bbox = item[config.BBOX_KEY]
                bboxes.append(bbox[0] + 0.5 * bbox[2])
                bboxes.append(bbox[1] + 0.5 * bbox[3])
                name_list.append(key)
        if len(bboxes) < 10:
            for i in range(10 - len(bboxes)):
                bboxes.append(0)
        if len(name_list) < 6:
            for i in range(6 - len(name_list)):
                name_list.append("z")

        inputs = np.array(pose + main_pos + bboxes)
        if labels["target"] in name_list:
            label = name_list.index(labels["target"])
        else:
            label = 0

        sample = {"inputs": inputs,
                  "labels": label,
                  "frame": labels["frame"],
                  "name_label": labels["target"],
                  "name": labels["name"],
                  "names_list": name_list,
                  "video": labels["video"]}

        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomPermutations(object):
    """

    """

    def __init__(self):
        return

    def __call__(self, sample):
        names_list = sample["names_list"]
        idxs = np.array([idx for idx in range(1, len(names_list))])
        np.random.shuffle(idxs)
        idxs = list(idxs)
        sample["names_list"] = [sample["names_list"][0]] + [sample["names_list"][idx] for idx in idxs]
        if sample["labels"] > 0:
            sample["labels"] = idxs.index(sample["labels"])+1
        new_input = []
        for i in idxs:
            new_input = np.concatenate((new_input,sample["inputs"][3+2*i:5+2*i]))
        sample["inputs"] = np.concatenate((sample["inputs"][:5], new_input), axis=0)
        return sample


class RandomTranslation(object):
    """

    """

    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, sample):
        x_shift = np.random.randint(-10, 10)
        y_shift = np.random.randint(-10, 10)
        inputs = sample["inputs"]
        for i in range(3, len(inputs), 2):
            if inputs[i] > 0 and inputs[i+1] > 0:
                inputs[i] += x_shift
                inputs[i+1] += y_shift

        sample["inputs"] = inputs
        return sample


class Normalization(object):
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, sample):

        inputs = sample["inputs"]
        inputs[:3] = (inputs[:3]) / 360
        inputs[3:] = inputs[3:] / self.img_size
        sample["inputs"] = inputs
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        inputs = sample["inputs"]
        sample["inputs"] = torch.from_numpy(inputs).type(torch.FloatTensor)
        return sample


if __name__ == "__main__":
    # dataset = FoADataset("data/labels/test_dataset_i.csv", "data/inputs",
    #                      transform=transforms.Compose([Normalization(), ToTensor()]))

    dataset = FoADataset("data/labels/test_test.csv", "data/inputs",
                         transform=transforms.Compose([Normalization(), ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    total = 0
    count = 0
    for i_batch, sample in enumerate(dataloader):
        break
        # for l in sample["name_label"]:
        #     if l == "z":
        #         count += 1
        #     total += 1
    # print(count / total)
