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
        #print(self.dataset)
        data = self.dataset[idx]
        #print(os.path.join(os.getcwd(), data[0]))
        
        # scans are loaded dynamically because I cant fit the entire dataset in RAM
        mat = nib.load(data[0])
        #mat = torch.Tensor(mat.get_fdata())
        mat = transform.resize(torch.Tensor(mat.get_fdata()), self.data_dim)

        
        # hacky fix to corner case
        #print("####", idx)
        return mat, torch.Tensor([float(data[1])])

        #return mat, torch.Tensor(int(data[1]))

class BaseParser:
    def __init__(self, data_dim):
        self.data_dim = data_dim
        self.subsets = []

    # create the dataset splits and shuffle them
    def create_dataset(self, splits, directory, extension = ".nii.gz"):
        dataset = []

        # walk all files in the dataset, and store the path and score
        for (root, dirs, files) in os.walk(os.path.join('ADNI', directory)):
            for file in files:
                if file[-len(extension):] == extension:
                    score = self.create_ground_truth(file)

                    if score is not None:
                        dataset.append((os.path.join(root, file), score))

        random.seed(263)
        random.shuffle(dataset)

        print("Dataset Length:", len(dataset))

        if round(sum(splits), 5) != 1.0:
            raise Exception("Dataset splits does not sum to 1")

        minIdx, maxIdx = 0, 0

        # split the dataset into the specified chunks
        for idx, split in enumerate(splits):
            chunk = int(len(dataset) * split)
            maxIdx += chunk

            subset = dataset[minIdx:maxIdx]

            random.seed(263)
            random.shuffle(subset)

            self.subsets.append(TorchLoader(np.array(subset), self.data_dim))

            minIdx += chunk
        #self.subsets = np.array(self.subsets)

    def get_subset(self, idx):
        return self.subsets[idx]

    def get_subset_array(self, idx):
        return self.subsets[idx].dataset

    def get_subset_length(self, idx):
        return len(self.subsets[idx])

    def create_ground_truth(self, filename):
        raise NotImplementedError("create_ground_truth not implemented")
