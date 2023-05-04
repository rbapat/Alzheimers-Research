from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import random
import torch
import os

from dataset_helper import create_class_dict

class SingleTPLoader(Dataset):
    def __init__(self, dataset, c):
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

class MultiTPLoader(Dataset):
    def __init__(self, dataset, c):
        self.c = c

        if c.LOAD_PATHS:
            self.dataset = dataset
        else:
            self.dataset = []
            for volume_paths, clin_vars, dx in dataset:
                # convert the .nii path to the .npy path I saved
                npy_paths = [path.replace(c.DATASET_PATH, c.EMBEDDING_PATH).replace('.nii', '.npy') for path in volume_paths]

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
        if self.c.LOAD_PATHS:
            volume_paths, clin_vars, dx = self.dataset[idx]

            return volume_paths, torch.Tensor(clin_vars), torch.LongTensor([dx])
        else:
            return self.dataset[idx]

# Main class to handle parsing the entire dataset and creating the specified data loaders
class DataParser:
    def __init__(self, c):

        # get all patients that convert from MCI to AD with the following clin vars, 3 timepoints 6 months apart
        
        paths, df, type_dict = create_class_dict(c.DATASET_PATH, c.CLIN_VARS, c.VISIT_DELTA, c.NUM_VISITS)
        if c.OPERATION & c.SINGLE_TIMEPOINT:
            dataset = self.create_single_tp_loader(df, type_dict, paths, c)
            dataset_loader = SingleTPLoader
        elif c.OPERATION & c.LONGITUDINAL:
            dataset = self.create_multi_tp_loader(df, type_dict, paths, c)
            dataset_loader = MultiTPLoader
        else:
            raise RuntimeError(f"Unknown operation: {c.OPERATION}")

        self.loaders, self.lengths = self.create_dataset(dataset, dataset_loader, c)

    def create_single_tp_loader(self, df, type_dict, paths, c):
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
                    data.append(cand[c.CLIN_VARS].values[0])
                    break
        
        # normalize each clinical variable to have zero mean and unit variance
        data = np.array(data)
        mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)
        data = np.apply_along_axis(lambda row: (row - mean) / std, 1, data)

        # A little hacky, but makes sure each group has the same number of patients in the created dataset (omits extra patients)
        dx_cap = c.DX_CAP
        counter = [min(counter[:dx_cap])] * 3
        
        # Create the actual dataset, omitting patients with MCI if dx_cap == 2 and making sure there are the same number of patients with AD/NC/MCI
        result = []
        for (dx, image_id), data_row in zip(selected, data):
            dx_idx = gts.index(dx)

            if dx_idx < dx_cap and counter[dx_idx] > 0:
                result.append((paths[int(image_id)], data_row, dx_idx))
                counter[dx_idx] -= 1

        return result

    def create_multi_tp_loader(self, df, type_dict, paths, c):
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
            for v in rows[c.CLIN_VARS].values:
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
    def create_dataset(self, dataset, loader_class, c):
        print(f"Processing dataset length {len(dataset)}")
        random.shuffle(dataset)

        min_idx, max_idx = 0, 0
        subsets, lengths, real_lens = [], [], []

        for split_ratio in c.SPLITS:
            chunk = int(len(dataset) * split_ratio)

            max_idx += chunk
            loader = loader_class(dataset[min_idx:max_idx], c)
            subsets.append(DataLoader(loader, batch_size = c.BATCH_SIZE, shuffle = True))
            lengths.append(max_idx - min_idx)
            real_lens.append(len(loader.dataset))
            min_idx += chunk

        print("Created the following subsets:")
        for idx, sub in enumerate(subsets):
            print(f"{idx}:\t{lengths[idx]} (augmented to {real_lens[idx]})")
        print()

        return subsets, real_lens


