import skimage.transform as transform
from torch.utils.data import Dataset
from extractor import Extractor
import nibabel as nib
import pandas as pd
import numpy as np
import random
import torch
import os

from util import data_utils

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
    def __init__(self, csv_path, data_dim, num_output, splits = [0.8, 0.2]):
        self.data_dim = data_dim
        self.num_output = num_output

        self.parse_csv(csv_path)
        self.create_dataset(splits)        

    def get_image_id(self):
        sizes = min(len(self.cn), len(self.mci), len(self.ad))
        return [[i for i in j[0:sizes]["Image ID"]] for j in [self.cn, self.mci, self.ad]]

    def augment_data(self, batch, dataset):
        return dataset

        master_dataset = []

        for idx, data in enumerate(dataset):

            augmented = nib.load(data[0])
            mat = augmented.get_fdata()

            mat += np.random.normal(0.0, np.sqrt(0.1), size = mat.shape)

            augmented = nib.Nifti1Image(mat, augmented.affine)
            path = os.path.join("Augmented", '%d_%d.nii' % (batch, idx))
            nib.save(augmented, path)

            master_dataset.append(data)
            master_dataset.append((path, data[1]))

        return master_dataset

    def create_dataset(self, splits):
        if not os.path.exists('Original'):
            path = os.path.join(os.getcwd(), 'Original')
            raise Exception("dataset is not located in directory %s" % path)

        if not os.path.exists('Processed'):
            print("Warning: Skull Stripped dataset does not exist, stripping right now.")
            self.write_stripped_dataset()

        dirs = ["CN", "AD", "MCI"]
        # ** This was used when training on just CN and MCI **
        #dirs = ["CN", "MCI", "AD"]

        dataset = []

        for idx in range(self.num_output):
            dataset += self.parse_directory(dirs[idx], idx)

        seed = 263
        random.Random(seed).shuffle(dataset)

        if sum(splits) != 1.0:
            raise Exception("Dataset splits does not sum to 1")

        self.subsets = []
        minIdx, maxIdx = 0, 0

        for idx, split in enumerate(splits):
            chunk = int(len(dataset) * split)
            maxIdx += chunk

            subset = self.augment_data(idx, dataset[minIdx:maxIdx])
            random.Random(seed).shuffle(subset)

            self.subsets.append(TorchLoader(subset, self.data_dim, self.num_output))

            minIdx += chunk

    def get_loader(self, idx):
        return self.subsets[idx]

    def get_set_length(self, idx):
        return len(self.subsets[idx])

    def parse_directory(self, directory, cid):
        directory = os.path.join("Processed", directory)
        subset = []

        for (root, dirs, files) in os.walk(directory):
            for file in files:
                if file[-4:] == '.nii':
                    subset.append((os.path.join(root, file), cid))

        return subset

    def write_stripped_dataset(self, keep_prob = 0.5):
        extractor = Extractor()

        create = True
        if os.path.exists('Processed'):
            create = False

        idx = [0, 0, 0]
        for (root, dirs, files) in os.walk('Original'):
            for file in files:
                if file[-4:] == '.nii':
                    print("Stripping file", file)

                    mat = nib.load(os.path.join(root, file))

                    prob = extractor.run(mat.get_fdata())
                    mask = prob > keep_prob

                    brain = mat.get_fdata()[:]
                    brain[~mask] = 0
                    brain = data_utils.crop_scan(brain)

                    '''
                    if '\\CN\\' in root:
                        name = os.path.join('CN', 'CN_%d.nii' % idx[0])
                        idx[0] += 1
                    elif '\\MCI\\' in root:
                        name = os.path.join('MCI', 'MCI_%d.nii' % idx[1])
                        idx[1] += 1
                    elif '\\AD\\' in root:
                        name = os.path.join('AD', 'AD_%d.nii' % idx[2])
                        idx[2] += 1

                    path = os.path.join('Processed', name)
                    '''
                        
                    path = os.path.join(root.replace('Original', 'Processed'), file)
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))

                    brain = nib.Nifti1Image(brain, mat.affine)
                    nib.save(brain, path)


    def parse_csv(self, csv_path):
        df = pd.read_csv(csv_path)

        df = df[np.logical_or(df["Description"] == 'MPR; GradWarp; B1 Correction; N3 <- MPRAGE', df["Description"] == 'MPR; GradWarp; B1 Correction; N3 <- MP-RAGE')]
        # ** This was used when training on just CN and MCI **
        #df = df[np.logical_or.reduce((df["Description"] == 'MPR; GradWarp; B1 Correction <- MPRAGE', df["Description"] == 'MPR; GradWarp; B1 Correction <- MP-RAGE', df["Description"] == 'MT1; GradWarp; N3m <- MPRAGE'))]

        df = df.drop_duplicates(subset="Subject ID", keep="first")

        self.cn = data_utils.get_group(df, "Research Group", "CN")
        self.mci = data_utils.get_group(df, "Research Group", "MCI")
        self.ad = data_utils.get_group(df, "Research Group", "AD")

        
