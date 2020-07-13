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

        class_tuple = data[1]
        non_imaging_inputs = torch.tensor(class_tuple[1:])
        #(cid, is_male, not is_female, *(class_data[i].values[0] for i in ["MMSE Total Score","Global CDR","FAQ Total Score"]))

        one_hot = np.zeros(self.num_output)
        one_hot[class_tuple[0]] = 1

        return mat, one_hot, non_imaging_inputs

class DataParser:
    def __init__(self, csv_path, data_dim, num_output, splits = [0.8, 0.2]):
        self.data_dim = data_dim
        self.num_output = num_output

        self.parse_csv(csv_path)
        self.create_dataset(splits)        

    def create_dataset(self, splits):
        if not os.path.exists('Original'):
            path = os.path.join(os.getcwd(), 'Original')
            raise Exception("dataset is not located in directory %s" % path)

        if not os.path.exists('Processed'):
            print("Warning: Skull Stripped dataset does not exist, stripping right now.")
            self.write_stripped_dataset()

        dataset = []
        for (root, dirs, files) in os.walk("Processed"):
            for file in files:
                if file[-4:] == '.nii':
                    labels = self.get_class_data(file)

                    if labels[0] > self.num_output - 1:
                        continue

                    dataset.append((os.path.join(root, file), labels))

        seed = 263
        random.Random(seed).shuffle(dataset)

        if sum(splits) != 1.0:
            raise Exception("Dataset splits does not sum to 1")

        self.subsets = []
        minIdx, maxIdx = 0, 0

        for idx, split in enumerate(splits):
            chunk = int(len(dataset) * split)
            maxIdx += chunk

            subset = dataset[minIdx:maxIdx]
            random.Random(seed).shuffle(subset)

            self.subsets.append(TorchLoader(subset, self.data_dim, self.num_output))

            minIdx += chunk

    def get_loader(self, idx):
        return self.subsets[idx]

    def get_set_length(self, idx):
        return len(self.subsets[idx])

    def get_class_data(self, filename):
        start_idx = filename.rindex('_') + 2;
        image_id = filename[start_idx:-4]

        class_data = self.df[self.df['Image ID'] == int(image_id)]

        cid = ["CN", "AD", "MCI"].index(class_data['Research Group'].values[0])

        is_male = int(class_data["Sex"].values[0] == "M")
        is_female = int(not is_male)

        labels = [cid, is_male, is_female, *(class_data[i].values[0] for i in ["MMSE Total Score","FAQ Total Score"])] #["MMSE Total Score","FAQ Total Score"]
        
        labels[3] /= 30.0
        labels[4] /= 30.0

        return labels

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
                        
                    path = os.path.join(root.replace('Original', 'Processed'), file)
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))

                    brain = nib.Nifti1Image(brain, mat.affine)
                    nib.save(brain, path)

    def parse_csv(self, csv_path):
        df = pd.read_csv('feature_rich_dataset.csv')

        drops = []
        for row in df.itertuples():
            for i in [2, 8, 11]:
                if row[i] != row[i]:
                    drops.append(row[0])
                    break
        df = df.drop(drops)

        self.df = df[np.logical_or(df["Description"] == 'MPR; GradWarp; B1 Correction; N3 <- MPRAGE', df["Description"] == 'MPR; GradWarp; B1 Correction; N3 <- MP-RAGE')]