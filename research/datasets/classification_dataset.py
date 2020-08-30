import skimage.transform as transform
from torch.utils.data import Dataset
from research.util.extractor import Extractor
import nibabel as nib
import pandas as pd
import numpy as np
import random
import torch
import os

from research.util.util import data_utils

# Custom implementation of PyTorch's default data loader
class TorchLoader(Dataset):
    def __init__(self, dataset, data_dim, num_output):
        self.dataset = dataset
        self.num_output = num_output
        self.data_dim = data_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # scans are loaded dynamically because I cant fit the entire dataset in RAM
        mat = nib.load(data[0])
        mat = transform.resize(torch.Tensor(mat.get_fdata()), self.data_dim)

        mat = 255 * (mat - mat.min())/(mat.max() - mat.min())

        one_hot = np.zeros(self.num_output)
        one_hot[data[1]] = 1

        return mat, one_hot

class DataParser:
    def __init__(self, csv_path, data_dim, num_output, splits = [0.8, 0.2]):
        self.data_dim = data_dim
        self.num_output = num_output

        self.df = pd.read_csv(csv_path)
        self.create_dataset(splits)        

    def create_dataset(self, splits):
        # original dataset isnt found, throw error
        if not os.path.exists(os.path.join('ADNI', 'Original')):
            path = os.path.join(os.getcwd(), 'ADNI', 'Original')
            raise Exception("dataset is not located in directory %s" % path)

        # skull strip and crop all the frames if it hasn't been done already
        # only checks if directory exists so pls dont have an empty or half-finished directory
        if not os.path.exists(os.path.join('ADNI', 'Processed')):
            self.write_stripped_dataset()

        dataset = []
        for (root, dirs, files) in os.walk(os.path.join('ADNI', 'Processed')):
            for file in files:
                if file[-4:] == '.nii':
                    cid = self.get_class_data(file)

                    # useful if we only want CN/AD or we want CN/AD/MCI
                    if cid < self.num_output:
                        dataset.append((os.path.join(root, file), cid))

        random.shuffle(dataset)

        if sum(splits) != 1.0:
            raise Exception("Dataset splits does not sum to 1")

        self.subsets = []
        minIdx, maxIdx = 0, 0

        # split the dataset into the specified chunks
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

        subj = self.df[self.df["Image ID"] == image_id]
        cid = ["CN", "AD", "MCI"].index(subj.iloc[0]["Research Group"])
        
        return cid

    def write_stripped_dataset(self, keep_prob = 0.5):
        #deepbrain skull stripper
        extractor = Extractor()

        for (root, dirs, files) in os.walk(os.path.join('ADNI', 'Original')):
            for file in files:
                if file[-4:] == '.nii':

                    mat = nib.load(os.path.join(root, file))
                    try:
                        prob = extractor.run(mat.get_fdata())
                    except Exception as e:
                        print("Error stripping skull:", str(e))
                        
                    mask = prob > keep_prob

                    brain = mat.get_fdata()[:]
                    brain[~mask] = 0
                    brain = data_utils.crop_scan(brain)
                        
                    path = os.path.join(root.replace('Original', 'Processed'), file)
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))

                    brain = nib.Nifti1Image(brain, mat.affine)
                    nib.save(brain, path)

    # used for parsing non-imaging data as well, not in use right now
    '''
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

    def get_data_list(self):
        df = pd.read_csv('ADNIMERGE.csv')
        images = pd.read_csv('adni_images.csv')

        #df = self.strip_nulls(df, "PTID", "AGE", "DX", "PTGENDER", "CDRSB", "MMSE", "FAQ", "IMAGEUID", "M")
        df = self.strip_nulls(df, "PTID", "DX", "IMAGEUID", "M")
        idx_tuple = self.get_idx_tup(list(df), "PTID", "AGE", "DX", "PTGENDER", "CDRSB", "MMSE", "FAQ", "IMAGEUID", "M")

        pt_list = []
        for i in df["PTID"].unique():
            patient_data = df[df["PTID"] == i].values
            sublist = []

            for patient in patient_data:
                patient_info = list(patient[idx_tuple])
                if len(images[images["Image ID"] == patient_info[6]]) == 0:
                    continue
            
                patient_info.append(images[images["Image ID"] == int(patient_info[6])]["Description"].values[0])

                hotlist = ["GradWarp", "B1 Correction", "N3"]
                if sum([i in patient_info[-1] for i in hotlist]) != len(hotlist):
                    continue

                patient_info[6] = int(patient_info[6])
                sublist.append(patient_info)

            if len(sublist) > 0:
                pt_list.append(sorted(sublist, key = lambda x: x[8]))

        #id, age, gender, cdrsb, mmse, faq, image id, dx, m, description
        self.data_list = []
        for pt in pt_list:
            for visit in pt:
                self.data_list.append(visit)
                break        
    '''