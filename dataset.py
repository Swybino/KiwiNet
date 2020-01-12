import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from side_script.data_processor import DataProcessor
import numpy as np
import config
from torchvision import transforms, utils
from random import random


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
        output = labels.to_dict()
        video = output.pop("video", None)
        frame = int(output.pop("frame", None))
        inputs = []
        file_path = os.path.join(self.root_dir, '%s_%s.json' %(video, frame))
        with open(file_path, 'r') as f:
            frame_data = json.load(f)

        for key in output.keys():
            name = config.kids_code[video][key]
            if name in frame_data:
                data = frame_data[name]
                bbox = data[config.BBOX_KEY]
                confidence = data[config.CONFIDENCE_KEY]
                pose = [x * confidence for x in data[config.POSE_KEY]]
                inputs.append(bbox + pose)
            else:
                inputs.append([0 for _ in range(7)])
        inputs = np.array(inputs)
        sample = {"inputs": inputs, "result": output, "frame": frame}

        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomPermutations(object):
    """

    """

    def __init__(self):
        return

    def __call__(self, sample):
        out = sample["result"]
        keys = list(out.keys())
        idxs = np.array([idx for idx in range(len(keys))])
        np.random.shuffle(idxs)
        sample["inputs"] = np.array([sample["inputs"][i] for i in idxs])
        sample["result"] = {keys[i]: out[keys[i]] for i in idxs}

        return sample


class RandomDelete(object):
    """

    """

    def __init__(self):
        return

    def __call__(self, sample):
        proba = 0.1
        keys = list(sample["result"].keys())
        for idx in range(len(keys)):
            if random() > 1 - proba:
                sample["result"][keys[idx]] = "z"
                sample["inputs"][idx][-3:] *= 0
                return sample
        return sample


class Binarization(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        result_data = sample["result"]
        keys = list(result_data.keys())
        out_data = []
        for idx, k in enumerate(keys):
            if result_data[k] in keys:
                out_data.append([1, keys.index(result_data[k])])
            else:
                out_data.append([0, idx])
        sample['labels'] = np.array(out_data)
        # return sample
        return {"inputs": sample["inputs"], 'labels': np.array(out_data)}


class Normalization(object):
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, sample):
        inputs = sample["inputs"]
        inputs[:, -3:] = (inputs[:, -3:] + 180) / 360
        inputs[:, :-3] = inputs[:, :-3] / self.img_size
        sample["inputs"] = inputs
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        inputs, output = sample["inputs"], sample['labels']
        sample["inputs"] = torch.from_numpy(inputs).type(torch.FloatTensor)
        sample['labels'] = torch.from_numpy(output).type(torch.LongTensor)
        # sample['inputs'] = sample['inputs']
        # sample['labels'] = sample['labels']
        return sample


if __name__ == "__main__":
    dataset = FoADataset("data/labels", "data/inputs", transform=transforms.Compose(
        [RandomDelete(), RandomPermutations(), Binarization(size=6), Normalization(img_size=640), ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched, "##########",
              sep="\n")
        break
