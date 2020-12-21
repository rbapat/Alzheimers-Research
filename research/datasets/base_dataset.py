import skimage.transform as transform
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import random
import torch
import os

# Custom implementation of PyTorch's default data loader
class TorchLoader(Dataset):
    def __init__(self, dataset, data_dim):
        self.dataset = dataset
        self.data_dim = data_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # scans are loaded dynamically because I cant fit the entire dataset in RAM
        mat = nib.load(data[0])
        mat = transform.resize(torch.Tensor(mat.get_fdata()), self.data_dim)

        # hacky fix to corner case
        if type(data[1]) == type(0.0):
            return mat, torch.Tensor([data[1]])

        return mat, torch.Tensor(data[1])

class BaseParser:
    def __init__(self, data_dim):
        self.data_dim = data_dim
        self.subsets = []

    def create_dataset(self, splits, directory, extension = ".nii.gz"):
        dataset = []
        for (root, dirs, files) in os.walk(os.path.join('ADNI', directory)):
            for file in files:
                if file[-len(extension):] == extension:
                    score = self.create_ground_truth(file)

                    if score is not None:
                        dataset.append((os.path.join(root, file), score))

        random.shuffle(dataset)

        if round(sum(splits), 5) != 1.0:
            raise Exception("Dataset splits does not sum to 1")

        minIdx, maxIdx = 0, 0

        # split the dataset into the specified chunks
        for idx, split in enumerate(splits):
            chunk = int(len(dataset) * split)
            maxIdx += chunk

            subset = dataset[minIdx:maxIdx]
            random.shuffle(subset)

            self.subsets.append(TorchLoader(subset, self.data_dim))

            minIdx += chunk

    def get_subset(self, idx):
        return self.subsets[idx]

    def get_subset_length(self, idx):
        return len(self.subsets[idx])

    def create_ground_truth(self, filename):
        raise NotImplementedError("create_ground_truth not implemented")