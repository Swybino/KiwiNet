import json
import os
from random import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils.utils
import config
import matplotlib.image as mpimg
import cv2


class FoADataset(Dataset):
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
        # Inputs
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

        # Labels
        if labels["target"] in name_list:
            label = name_list.index(labels["target"])
        else:
            label = 0

        # Eye Image
        confidence = frame_data[labels["name"]][config.CONFIDENCE_KEY]
        if confidence > 0:
            path = os.path.join(config.eye_img_root,
                                "%s_%s_%s.jpg" % (labels["video"], labels["frame"], labels["name"]))
            if os.path.exists(path):
                eye_img = np.array(mpimg.imread(path))
            else:
                landmarks = frame_data[labels["name"]][config.LANDMARKS_KEY]
                roll = frame_data[labels["name"]][config.POSE_KEY][2]
                eye_img = utils.utils.get_eye_image_from_video(labels["video"], labels["frame"], landmarks, roll)
                try :
                    eye_img_save = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(path, eye_img_save)
                except:
                    print("An exception occurred")
                    print(labels["video"], labels["frame"], labels["name"])
                    eye_img = np.zeros((224, 224, 3))
        else:
            eye_img = np.zeros((224, 224, 3))

        sample = {"inputs": inputs,
                  "labels": label,
                  "eye_img": eye_img,
                  "frame": labels["frame"],
                  "name_label": labels["target"],
                  "names_list": name_list,
                  "video": labels["video"],
                  "positions": torch.Tensor(main_pos + bboxes),
                  "name": labels["name"]}

        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomPermutations(object):
    """
    """

    def __init__(self):
        return

    def __call__(self, sample):
        out = sample["results"]
        keys = list(out.keys())
        idxs = np.array([idx for idx in range(len(keys))])
        np.random.shuffle(idxs)
        sample["inputs"] = np.array([sample["inputs"][i] for i in idxs])
        sample["results"] = {keys[i]: out[keys[i]] for i in idxs}
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
            inputs[i] += x_shift
            inputs[i + 1] += y_shift

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
        sample['eye_img'] = sample['eye_img'] / 255
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        inputs, output, eye_img = sample["inputs"], sample['labels'], sample['eye_img']
        sample["inputs"] = torch.from_numpy(inputs).type(torch.FloatTensor)
        eye_img = eye_img.transpose((2, 0, 1))
        sample['eye_img'] = torch.from_numpy(eye_img).type(torch.FloatTensor)
        return sample


if __name__ == "__main__":
    # dataset = FoADataset("data/labels/test_dataset_i.csv", "data/inputs",
    #                      transform=transforms.Compose([Normalization(), ToTensor()]))

    # dataset = FoADataset("data/labels/test_labels_frame_patches.csv", "data/inputs",
    #                      transform=transforms.Compose([ToTensor()]))

    dataset = FoADataset("data/labels/test_labels_frame_patches.csv", "data/inputs",
                         transform=transforms.Compose([ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=2,
                            shuffle=True, num_workers=0)

    for i in range(8932, 10000):
        print(i)
        dataset.__getitem__(i)

    # total = 0
    # count = 0
    # for i_batch, sample in enumerate(dataloader):
    #     print("#####", i_batch, sample["video"], sample["frame"])
    #
    # print(count / total)
