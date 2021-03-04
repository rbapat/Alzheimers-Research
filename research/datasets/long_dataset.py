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

        mats = []
        for trip in data[0]:
            mat = nib.load(trip[0])
            mat = transform.resize(torch.Tensor(mat.get_fdata()), self.data_dim)
            mats.append(mat)

        one_hot = np.zeros(2)
        one_hot[data[1]] = 1

        '''
        # hacky fix to corner case
        if type(data[1]) == type(0.0):
            return mat, torch.Tensor([data[1]])

        print(data[1])
        return mat, torch.Tensor(data[1])
        '''

        return torch.Tensor(mats), one_hot

# MONTH, ISCONVERTER, DIAGNOSIS

class DataParser:
    def __init__(self, data_dim, splits = [0.8, 0.2]):
        self.data_dim = data_dim
        self.subsets = []

        self.extension = '.nii.gz'

        self.create_dataset(splits, self.init_df())

    def init_df(self):
        keys = ["VISCODE", "PTID", "IMAGEUID", "LDELTOTAL", "MMSE", "CDRSB", "mPACCdigit", "mPACCtrailsB", "DX"]
        self.df = pd.read_csv('ADNIMERGE.csv', low_memory = False).dropna(subset = keys) 

        mci_ad, mci_mci = set(), set()
        for scan in self.df.iloc:
            if (scan["DX_bl"] == "LMCI" or scan["DX_bl"] == "EMCI") and scan["DX"] == "Dementia":
                mci_ad.add(scan["PTID"])

            d = np.unique(self.df[self.df["PTID"] == scan["PTID"]]["DX"].values)

            if len(d) == 1 and d[0] == "MCI":
                mci_mci.add(scan["PTID"])

        im_dict = {}
        #for (root, dirs, files) in os.walk('/home/rohan/Alzheimers-Research/research/ADNI/AllData_FSL'):
        for (root, dirs, files) in os.walk('/home/rohan_bapat/Alzheimers-Research/research/ADNI/AllData_FSL'):
            for file in files:
                if file[-7:] == '.nii.gz':
                    image_id = int(file[file.rindex('_') + 2:-7])
                    im_dict[image_id] = os.path.join(root, file)

        month_df = self.df[~(self.df["Month"] > 24)]

        dataset = []
        for ptid in mci_ad:
            scans = month_df[month_df["PTID"] == ptid].sort_values(by=['Month'])
            months = scans["Month"].values

            if 0 in months and 12 in months and 24 in months:
                data = scans[["PTID", "Month", "DX", "IMAGEUID"]] 

                triplet = []
                for item in data.values:
                    try:
                        path = im_dict[int(item[3])]
                    except KeyError:
                        print("Failed", int(item[3]))
                        continue
                    dx = ["CN", "MCI", "Dementia"].index(item[2])
                    triplet.append((path, (item[1], dx)))

                if len(triplet) == 3:
                    dataset.append((triplet, 1))

        cvter = len(dataset)

        for ptid in mci_mci:
            scans = month_df[month_df["PTID"] == ptid].sort_values(by=['Month'])
            months = scans["Month"].values

            if 0 in months and 12 in months and 24 in months:
                data = scans[["PTID", "Month", "DX", "IMAGEUID"]] 

                triplet = []
                for item in data.values:
                    try:
                        path = im_dict[int(item[3])]
                    except KeyError:
                        print("Failed", int(item[3]))
                        continue

                    dx = ["CN", "MCI", "Dementia"].index(item[2])
                    triplet.append((path, (item[1], dx)))

                if len(triplet) == 3:
                    dataset.append((triplet, 0))

        print("%d converters, %d nonconverters" % (cvter, len(dataset) - cvter))
        return dataset
            
    def create_dataset(self, splits, data):
        dataset = data

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
            random.shuffle(subset)

            self.subsets.append(TorchLoader(subset, self.data_dim))

            minIdx += chunk

    def get_subset(self, idx):
        return self.subsets[idx]

    def get_subset_length(self, idx):
        return len(self.subsets[idx])
