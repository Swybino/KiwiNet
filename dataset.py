import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from side_script.data_processor import DataProcessor
import numpy as np
import config

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir, video_title, transform=None):
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
        self.data_processor = DataProcessor(root_dir, video_title)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = self.labels.iloc[idx]
        output = labels.to_dict()

        input = [np.array(self.data_processor.get_item(labels.name, name, config.POSE_KEY))
                 * self.data_processor.get_item(labels.name, name, config.CONFIDENCE_KEY)
                 for name in output.keys()]

        sample = {"input": input, "out": output}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomPermutations(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        return

    def __call__(self, sample):
        out = sample["out"]
        keys = list(out.keys())
        idxs = np.array([idx for idx in range(len(keys))])
        np.random.shuffle(idxs)
        sample["input"] = np.array([sample["input"][i] for i in idxs])
        sample["out"] = {keys[i]: out[keys[i]] for i in idxs}

        return sample

class Binarization(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        return

    def __call__(self, sample):
        data = sample["out"]
        keys = data.keys()
        out_data = np.array([])

        for idx, k in enumerate(keys):
            out_data[idx] = [0 for _ in keys]
            if data[k] in keys:
                out_data[idx, keys.index(data[k])] = 1
            else:
                out_data[idx, idx] = 1

        sample["out"] = out_data
        return sample


