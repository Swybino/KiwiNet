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
from utils.video import Video
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
        landmarks = frame_data[labels["name"]][config.LANDMARKS_KEY]
        img = Video(os.path.join(config.video_root, "%s.MP4" % labels["video"]))[int(labels["frame"])]
        eye_img = self.get_eye_image(img, landmarks)
        eye_img = cv2.resize(eye_img, (224, 224))

        sample = {"inputs": inputs, "labels": label, "frame": labels["frame"], "name_label": labels["target"],
                  "names_list": name_list, "video": labels["video"], "positions": torch.Tensor(main_pos + bboxes),
                  "eye_img": eye_img}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_eye_image(self, img, landmarks):
        eye_bbox = [min(landmarks[0][42:48]),
                    min(landmarks[1][42:48]),
                    max(landmarks[0][42:48])-min(landmarks[0][42:48]),
                    max(landmarks[1][42:48])-min(landmarks[1][42:48])]
        eye_img, _ = utils.utils.crop_roi(img, eye_bbox, padding=10)
        return eye_img

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

    dataset = FoADataset("data/labels/labels.csv", "data/inputs",
                         transform=transforms.Compose([ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    total = 0
    count = 0
    for i_batch, sample in enumerate(dataloader):
        print("********",sample["eye_img"].size())
        for i, f in enumerate(sample["eye_img"]):

            cv2.imshow("img%s" %i, cv2.cvtColor(np.array(f), cv2.COLOR_BGR2RGB))
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(sample)
        break

    print(count / total)
