import skimage.transform as transform
from torch.utils.data import Dataset
from extractor import Extractor
import nibabel as nib
import pandas as pd
import numpy as np
import random
import torch
import os

from research.util.util import data_utils

# TODO

class TorchLoader(Dataset):
    def __init__(self, dataset, data_dim, num_output):
        self.dataset = dataset
        self.num_output = num_output
        self.data_dim = data_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        

        mat = nib.load(data[0])
        mat = transform.resize(torch.Tensor(mat.get_fdata()), self.data_dim)

        one_hot = np.zeros(self.num_output)
        one_hot[data[1]] = 1

        return mat, one_hot

class DataParser:
    def __init__(self, data_dim, num_output, splits = [0.8, 0.2]):
        self.data_dim = data_dim
        self.num_output = num_output

        self.create_dataset(splits)        

    def create_dataset(self, splits):
        downloaded_df = pd.read_csv('freesurfer_dataset.csv')

        if not os.path.exists(os.path.join('ADNI', 'FSOriginal')):
            path = os.path.join(os.getcwd(), 'ADNI', 'FSOriginal')
            raise Exception("dataset is not located in directory %s" % path)

        if not os.path.exists(os.path.join('ADNI', 'FSProcessed')):
            raise Exception('Couldn\'t find processed dataset')
            #self.write_stripped_dataset()

        data_split = [[], [], []]
        for (root, dirs, files) in os.walk(os.path.join('ADNI', 'FSProcessed')):
            for file in files:
                if file[-4:] == '.nii':
                    ptid = root[:root.rindex('/')]
                    ptid = ptid[:ptid.rindex('/')]
                    ptid = ptid[ptid.index('/'):]
                    ptid = ptid[ptid.index('/') + 1:]
                    ptid = ptid[ptid.index('/') + 1:]
                    ptid = ptid[:ptid.index('/')]

                    patient = downloaded_df[downloaded_df["Subject"] == ptid]

                    cid = ["MCI", "AD", "CN"].index(patient["Group"].values[0])

                    if cid < self.num_output:
                        data_split[cid].append((os.path.join(root, file), cid))


        min_len = min([len(i) for i in data_split[:self.num_output]])
        dataset = []
        for i in range(self.num_output):
            dataset += data_split[i][:min_len]

        random.shuffle(dataset)

        if sum(splits) != 1.0:
            raise Exception("Dataset splits does not sum to 1")

        self.subsets = []
        minIdx, maxIdx = 0, 0

        for idx, split in enumerate(splits):
            chunk = int(len(dataset) * split)
            maxIdx += chunk

            subset = dataset[minIdx:maxIdx]
            random.shuffle(subset)

            self.subsets.append(TorchLoader(subset, self.data_dim, self.num_output))

            minIdx += chunk

    def get_loader(self, idx):
        return self.subsets[idx]

    def get_set_length(self, idx):
        return len(self.subsets[idx])

    def get_class_data(self, filename):
        start_idx = filename.rindex('_') + 2;
        image_id = int(filename[start_idx:-4])

        target = None
        for data in self.data_list:
            if data[6] == image_id:
                target = data

        if target is None:
            return target

        #id, age, gender, cdrsb, mmse, faq, image id, dx, m, description
        cid = ["CN", "Dementia", "MCI"].index(target[7])

        is_male = int(target[2] == "M")
        is_female = int(not is_male)

        labels = [cid, is_male, is_female, target[3], target[4], target[5]]
        
        labels[3] /= 18.0        
        labels[4] /= 30.0
        labels[5] /= 30.0

        return labels

    def strip_nulls(self, df, *args):
        for i in args:
            df = df[~df[i].isnull()]
        return df

    def get_idx_tup(self, headers, *args):
        arr = []

        for i, x in enumerate(headers):
            if x in args:
                arr.append(i)

        return arr
