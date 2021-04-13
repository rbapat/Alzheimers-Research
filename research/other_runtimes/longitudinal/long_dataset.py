import skimage.transform as transform
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import random
import torch
import os

# Custom implementation of PyTorch's default data loader
class LongTorchLoader(Dataset):
    def __init__(self, dataset, data_dim):
        self.dataset = dataset
        self.data_dim = data_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # scans are loaded dynamically because I cant fit the entire dataset in RAM
        # ((path, month, score), idx)
        mats = []  
        for trip in data[0][:-1]:
            mat = nib.load(trip[0])
            mat = transform.resize(torch.Tensor(mat.get_fdata()), self.data_dim)
            mats.append(mat)

        scores = [i[2] for i in data[0][1:]]
        return torch.Tensor(mats), torch.Tensor(scores)

class SingleTorchLoader(Dataset):
    def __init__(self, dataset, data_dim):
        self.dataset = dataset
        self.data_dim = data_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # scans are loaded dynamically because I cant fit the entire dataset in RAM
        # ((path, month, score), idx)
        mats = nib.load(data[0])  
        mat = transform.resize(torch.Tensor(mat.get_fdata()), self.data_dim)

        return mat, torch.Tensor(data[2])

# MONTH, ISCONVERTER, DIAGNOSIS

class DataParser:
    def __init__(self, data_dim, splits = [0.8, 0.2], longitudinal = True):
        self.data_dim = data_dim
        self.long = longitudinal
        self.subsets = []

        self.extension = '.nii.gz'

        self.create_dataset(splits, self.init_df())

    def get_image_dict(self):
            im_dict = {}
            #for (root, dirs, files) in os.walk('/home/rohan/Alzheimers-Research/research/ADNI/AllData_FSL'):
            for (root, dirs, files) in os.walk('/home/rohan_bapat/Alzheimers-Research/research/ADNI/AllData_FSL'):
                for file in files:
                    if file[-7:] == '.nii.gz':
                        image_id = int(file[file.rindex('_') + 2:-7])
                        im_dict[image_id] = os.path.join(root, file)

            return im_dict

    def get_class_sets(self, df):
        cn_mci, mci_ad, cn_cn, mci_mci, ad_ad = set(), set(), set(), set(), set()
        for scan in df.iloc:
            if scan["DX_bl"] == "CN" and scan["DX"] == "MCI":
                cn_mci.add(scan["PTID"])
            elif (scan["DX_bl"] == "LMCI" or scan["DX_bl"] == "EMCI") and scan["DX"] == "Dementia":
                mci_ad.add(scan["PTID"])

            d = np.unique(df[df["PTID"] == scan["PTID"]]["DX"].values)

            if len(d) == 1 and d[0] == "CN":
                cn_cn.add(scan["PTID"])
            elif len(d) == 1 and d[0] == "MCI":
                mci_mci.add(scan["PTID"])
            elif len(d) == 1 and d[0] == "Dementia":
                ad_ad.add(scan["PTID"])

        return cn_mci, mci_ad, cn_cn, mci_mci, ad_ad

    def get_score_dict(self):
        df = pd.read_csv("scores.csv")
        data_dict = {}

        for index, row in df.iterrows():
            data_dict[int(row["IMAGEUID"])] = float(row["SCORE"])

        return data_dict

    def parse_cohort(self, dataset, idx, df, ptids, images, scores):
        for ptid in ptids:
            scans = df[df["PTID"] == ptid].sort_values(by=['Month'])
            months = scans["Month"].values

            data = scans[["PTID", "Month", "DX", "IMAGEUID"]] 
            if 0 in months and 12 in months and 24 in months:
                triplet = []
                for item in data.values:
                    if item[1] in [0, 12, 24]:
                        try:
                            path = images[int(item[3])]
                        except KeyError:
                            continue
                    
                        dx = scores[int(item[3])]
                        triplet.append((path, item[1], dx))

                if len(triplet) == 3 and triplet[0][2] != 2:
                    dataset[0].append((triplet, idx))
                else:
                    item = data.values[0]
                    dx = 0.0 #scores[int(item[3])]

                    try:
                        dataset[1].append((images[int(item[3])], item[1], dx))
                    except KeyError:
                        pass

            else:
                item = data.values[0]
                dx = 0.0 #scores[int(item[3])]

                try:
                    dataset[1].append((images[int(item[3])], item[1], dx))
                except KeyError:
                    pass

    def init_df(self):
        images = self.get_image_dict()
        scores = self.get_score_dict()

        keys = ["VISCODE", "PTID", "IMAGEUID", "LDELTOTAL", "MMSE", "CDRSB", "mPACCdigit", "mPACCtrailsB", "DX"]
        df = pd.read_csv('ADNIMERGE.csv', low_memory = False).dropna(subset = keys) 
        cn_mci, mci_ad, cn_cn, mci_mci, ad_ad = self.get_class_sets(df)

        # triplets, tertiary
        dataset = ([], [])

        self.parse_cohort(dataset, 0, df, cn_mci, images, scores)
        self.parse_cohort(dataset, 0, df, mci_ad, images, scores) # all of cvt
        self.parse_cohort(dataset, 1, df, cn_cn, images, scores)
        self.parse_cohort(dataset, 1, df, mci_mci, images, scores) # all (-1) noncvt
        self.parse_cohort(dataset, 1, df, ad_ad, images, scores) # only 1 noncvt

        if self.long:
            return dataset[0]
        else:
            return dataset[0]
            
    def create_dataset(self, splits, data):
        d = [0, 0]
        for triplet, idx in data:
            d[idx] += 1

        print(d[0], "Converters")
        print(d[1], "Non converters")

        random.shuffle(data)

        print("Dataset Length:", len(data))

        if round(sum(splits), 5) != 1.0:
            raise Exception("Dataset splits does not sum to 1")

        minIdx, maxIdx = 0, 0

        # split the dataset into the specified chunks
        for idx, split in enumerate(splits):
            chunk = int(len(data) * split)
            maxIdx += chunk

            subset = data[minIdx:maxIdx]
            random.shuffle(subset)

            if self.long:
                self.subsets.append(LongTorchLoader(subset, self.data_dim))
            else:
                self.subsets.append(SingleTorchLoader(subset, self.data_dim))

            minIdx += chunk

    def get_subset(self, idx):
        return self.subsets[idx]
    
    def get_num_outputs(self):
        return 2

    def get_subset_length(self, idx):
        return len(self.subsets[idx])
