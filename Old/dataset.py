from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import random
import torch
import os

from dataset_helper import create_class_dict

'''
Pytorch dataloader to handle single timepoint datasets. It just opens the .nii
scan, normalizes it, and returns it with the clinical variables and diagnosis

It handles all the batching internally so I just need to convert
the index of an item in the dataset into the actual data
'''
class SingleTPLoader(Dataset):
    def __init__(self, dataset, augment):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, clin_vars, dx = self.dataset[idx]

        mat = nib.load(path).get_fdata()
        mat = (mat - mat.min()) / (mat.max() - mat.min()) # min-max normalization

        return torch.Tensor(mat), torch.Tensor(clin_vars), torch.LongTensor([dx])


class MultiTPPathLoader(Dataset):
    def __init__(self, dataset, augment):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        volume_paths, clin_vars, dx = self.dataset[idx]

        return volume_paths, torch.Tensor(clin_vars), torch.LongTensor([dx])

'''
Pytorch dataloader to handle multi-timepoint datasets. Previously, I went through all the .nii scans and
fed it through a pretrained neural network. I then saved the convolutional features (output of the final convolutional layer)
as a .npy final to be used for the LSTM network. This is the data that this data loader is reading.

I read the 3 specified .npy files, and return it with the clinical variables at the 3 timepoints and the final diagnosis

Just as with the SingleTPLoader, pytorch handles all the batching internally.
'''
class MultiTPLoader(Dataset):
    def __init__(self, dataset, augment):
        self.dataset = []

        for volume_paths, clin_vars, dx in dataset:
            # convert the .nii path to the .npy path I saved
            npy_paths = [path.replace('/media/rohan/ThirdHardDrive/Combined_FSL_Old', '/home/rohan/Documents/Alzheimers/embeddings_1000').replace('.nii', '.npy') for path in volume_paths]

            # make sure all the files exist
            if False in [os.path.exists(path) for path in npy_paths]:
                raise RuntimeError("Please make sure to create the embeddings first.")

            # read .npy data and store it with clinical variables and final diagnoses
            # since the .npy data is much smaller than the .nii scan, I can just store the entire dataset in GPU memory, which makes this run much faster
            volumes = torch.Tensor(np.array([np.load(path).squeeze() for path in npy_paths]))
            self.dataset.append((volumes.cuda(), torch.Tensor(clin_vars).cuda(), torch.LongTensor([dx]).cuda()))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AugmentedMultiTPLoader(Dataset):
    def __init__(self, dataset, augment):
        self.dataset = []

        aug_ds = []
        if augment:
            for volume_paths, clin_vars, dx in dataset:
                aug_ds.append((volume_paths, clin_vars, dx))

                name = random.choice(['_rotated.nii', '_flipped.nii', '_noisy.nii'])
                new_volume_paths = [path.replace('.nii', name) for path in volume_paths]
                aug_ds.append((new_volume_paths, clin_vars, dx))

                '''
                for name in ['.nii', '_rotated.nii', '_flipped.nii', '_noisy.nii']:
                    new_volume_paths = [path.replace('.nii', name) for path in volume_paths]
                    aug_ds.append((new_volume_paths, clin_vars, dx))
                '''

            dataset = aug_ds

        for volume_paths, clin_vars, dx in dataset:
            # convert the .nii path to the .npy path I saved
            npy_paths = [path.replace('/media/rohan/ThirdHardDrive/Combined_FSL_Old', '/home/rohan/Documents/Alzheimers/embeddings_1000').replace('.nii', '.npy') for path in volume_paths]

            # make sure all the files exist
            if False in [os.path.exists(path) for path in npy_paths]:
                print(npy_paths)
                raise RuntimeError("Please make sure to create the embeddings first.")

            # read .npy data and store it with clinical variables and final diagnoses
            # since the .npy data is much smaller than the .nii scan, I can just store the entire dataset in GPU memory, which makes this run much faster
            volumes = torch.Tensor(np.array([np.load(path).squeeze() for path in npy_paths]))
            self.dataset.append((volumes.cuda(), torch.Tensor(clin_vars).cuda(), torch.LongTensor([dx]).cuda()))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Main class to handle parsing the entire dataset and creating the specified data loaders
class DataParser:
    def __init__(self, dataset_type, batch_size, path = '/media/rohan/ThirdHardDrive/Combined_FSL', splits = [(0.8, True), (0.2, False)]):
        random.seed(1)

        # Clinical variables that I want to include when training
        # Just experimenting with this, I find about 150 converters and 200 non-converters that have this info
        # Some patients don't have entries in all timepoints for this information so I just omit them
        clin_vars = ['MMSE', 'CDRSB', 'mPACCtrailsB', 'mPACCdigit']
        paths, df, type_dict = create_class_dict(path, clin_vars, 6, 3)

        # Types of datasets I'm supporting right now. 
        # SingleTPLoader = single timepoint classification (or regression) of AD/NC, can change a parameter to include MCI as well
        # MultiTPLoader = multi timepoint classification of whether or not a patient will convert from MCI to AD
        data_types = [
                        (self.create_3_classification, SingleTPLoader), 
                        (self.create_longitudinal, MultiTPLoader), 
                        (self.create_2_classification, SingleTPLoader), 
                        (self.create_longitudinal, MultiTPPathLoader), 
                        (self.create_longitudinal, AugmentedMultiTPLoader),
                        (self.create_3_classification, MultiTPPathLoader)
                    ]

        # parameter `dataset_type` determines which data loader (and therefore dataset) is created
        # 0 = AD/NC/MCI classification, 1 = longitudinal classification of conversion, 2 = AD/NC classification
        dataset_creator, dataset_loader = data_types[dataset_type]
        dataset = dataset_creator(df, type_dict, paths, clin_vars)

        self.loaders, self.lengths = self.create_dataset(dataset, splits, dataset_loader, batch_size)

    # creates the dataset for 3-way classification between AD/NC/MCI (based on dx_cap parameter)    
    def create_3_classification(self, df, type_dict, paths, clin_vars):
        return self.create_2_classification(df, type_dict, paths, clin_vars, dx_cap = 3)

    # creates the dataset for 2-way classification between AD and Normal (NC)
    def create_2_classification(self, df, type_dict, paths, clin_vars, dx_cap = 2):
        gts = ['CN', 'Dementia', 'MCI']
        selected, data = [], []
        
        counter = [0, 0, 0]
        for pt_key in type_dict:
            val, _ = type_dict[pt_key]
            if not val.startswith('CLASSIFICATION_'):
                continue



            pt_df = df[df["PTID"] == pt_key]
            candidates = [pt_df[pt_df["DX"] == dx] for dx in ['Dementia', 'MCI', 'CN']]
            
            # select one of the timepoints this patient has; prioritize dementia, then MCI, then normal diagnosis
            for cand in candidates:
                if len(cand) > 0:
                    selected.append(cand[["DX", "IMAGEUID"]].values[0])
                    counter[gts.index(selected[-1][0])] += 1
                    data.append(cand[clin_vars].values[0])
                    break
        
        # normalize each clinical variable to have zero mean and unit variance
        data = np.array(data)
        mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)
        data = np.apply_along_axis(lambda row: (row - mean) / std, 1, data)

        # A little hacky, but makes sure each group has the same number of patients in the created dataset (omits extra patients)
        counter = [min(counter[:dx_cap])] * 3
        
        # Create the actual dataset, omitting patients with MCI if dx_cap == 2 and making sure there are the same number of patients with AD/NC/MCI
        result = []
        for (dx, image_id), data_row in zip(selected, data):
            dx_idx = gts.index(dx)

            if dx_idx < dx_cap and counter[dx_idx] > 0:
                result.append((paths[int(image_id)], data_row, dx_idx))
                counter[dx_idx] -= 1

        return result

    # creates the dataset for longitudinal prediction of conversion from MCI to AD
    def create_longitudinal(self, df, type_dict, paths, clin_vars):
        selected, data = [], []
        seq_len = 3
        for pt_key in type_dict:
            val, ids = type_dict[pt_key] # ids is the IMAGEUID field of each timepoint of this patient
            pt_df = df[df["PTID"] == pt_key]

            if val.startswith('CLASSIFICATION_'):
                continue

            # get the rows in ADNIMERGE of this patient's 3 timepoints in order to get clinical variables
            rows = pt_df[pt_df["IMAGEUID"].isin(ids)].sort_values(by = ["Month"])
            dx = ['NON_CONVERTER', 'CONVERTER'].index(val)

            selected.append((rows["IMAGEUID"].values, dx))
            for v in rows[clin_vars].values:
                data.append(v)

        # normalize each clinical variable to have zero mean and unit variance
        data = np.array(data)
        mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)
        data = np.apply_along_axis(lambda row: (row - mean) / std, 1, data)

        # Create the actual dataset
        result = []
        for idx, (imids, dx) in enumerate(selected):
            volume_paths = [paths[int(image_id)] for image_id in imids]
            result.append((volume_paths, data[seq_len*idx:seq_len*idx+seq_len], dx))

        return result

    # Shuffle the dataset (deterministic because of random.seed earlier), split it into training and validation subsets, and create a pytorch DataLoader
    def create_dataset(self, dataset, splits, loader_class, in_batch_size):
        # print(f"Processing dataset length {len(dataset)}")
        random.shuffle(dataset)

        min_idx, max_idx = 0, 0
        subsets, lengths, real_lens = [], [], []

        for split_ratio, aug in splits:
            chunk = int(len(dataset) * split_ratio)

            max_idx += chunk
            loader = loader_class(dataset[min_idx:max_idx], augment = aug)
            subsets.append(DataLoader(loader, batch_size = in_batch_size, shuffle = True))
            lengths.append(max_idx - min_idx)
            real_lens.append(len(loader.dataset))
            min_idx += chunk

        print("Created the following subsets:")
        for idx, sub in enumerate(subsets):
            print(f"{idx}:\t{lengths[idx]} (augmented to {real_lens[idx]})")
        print()

        return subsets, real_lens


