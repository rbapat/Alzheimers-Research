from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import nibabel as nib
import pandas as pd
import numpy as np
import random
import torch
import os

from dataset_helper import create_class_dict

class LongitudinalDataset(Dataset):
    def __init__(self, vols, cvs, dxs):
        self.vols = vols
        self.cvs = cvs
        self.dxs = dxs

    def __len__(self):
        return len(self.vols)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.vols[idx], self.cvs[idx], self.dxs[idx]

class LongitudinalPathDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        volume_paths, clin_vars, dx = self.dataset[idx]

        return volume_paths, torch.Tensor(clin_vars), torch.LongTensor([dx])

class DatasetWrapper(Dataset):
    def __init__(self, idxList, origDataset):
        self.idxList = idxList;
        self.dataset = origDataset

    def __len__(self):
        return len(self.idxList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.dataset[self.idxList[idx]]

class DataParser:
    def __init__(self, settings):
        paths, df, type_dict = create_class_dict(settings.DATASET_PATH, settings.CLIN_VARS, settings.VISIT_DELTA, settings.NUM_VISITS)
        dataset = self.build_longitudinal_dataset(df, type_dict, paths, settings.CLIN_VARS)

        if settings.OUTER_SPLIT == 1:
            self.outer_skf = None
        else:
            self.outer_skf = StratifiedKFold(n_splits = settings.OUTER_SPLIT, shuffle = True)
        self.inner_skf = StratifiedKFold(n_splits = settings.INNER_SPLIT, shuffle = True)
        self.train_idx, self.test_idx, self.train_label = self.split_dataset(dataset)
        self.dataset = self.load_data(settings, dataset)
        self.settings = settings

        self.total_length = len(dataset)
        self.train_length = len(self.train_idx)
        self.test_length = len(self.test_idx)

        print("Dataset created!")
        print(f"Total size: {self.total_length}, split into train[{self.train_length}] and test[{self.test_length}]")

        if settings.OUTER_SPLIT == 1:
            for fold_idx, (train_set, val_set) in enumerate(self.cross_validation_set()):
                print(f"Fold {fold_idx} consists of train[{len(train_set.dataset)}] and val[{len(val_set.dataset)}]")
        else:
            for outer_fold_idx, (outer_train_split, test_set) in enumerate(self.cross_validation_set()):
                print(f"Outer fold {outer_fold_idx} consists of test[{len(test_set.dataset)}]")
                for inner_fold_idx, (train_set, val_set) in enumerate(self.cross_validation_set(outer_train_split)):
                    print(f"Inner fold {inner_fold_idx} consists of train[{len(train_set.dataset)}] and val[{len(val_set.dataset)}]")

    def build_longitudinal_dataset(self, df, type_dict, paths, clin_vars):
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
        dataset = []
        for idx, (imids, dx) in enumerate(selected):
            volume_paths = [paths[int(image_id)] for image_id in imids]
            dataset.append((volume_paths, data[seq_len*idx:seq_len*idx+seq_len], dx))

        return dataset

    def split_dataset(self, dataset):
        print(f"Processing dataset length {len(dataset)}")
        random.shuffle(dataset)

        # get list of indices, and list of ground truth
        X = list(range(len(dataset)))
        y = [item[2] for item in dataset]

        # split into train indices, test indices, train gt, test gt
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
        return np.array(X), np.array(X_test_idx), np.array(y)
        return np.array(X_train_idx), np.array(X_test_idx), np.array(y_train)

    def load_data(self, settings, dataset):
        if settings.LOAD_PATHS:
            return LongitudinalPathDataset(dataset)

        processed_data = []
        for volume_paths, clin_vars, dx in dataset:
            npy_paths = [path.replace(settings.DATASET_PATH, settings.EMBEDDING_PATH).replace('.nii', '.npy') for path in volume_paths]

            # make sure all the files exist
            if False in [os.path.exists(path) for path in npy_paths]:
                raise RuntimeError("Please make sure to create the embeddings first.")

            # read .npy data and store it with clinical variables and final diagnoses
            # since the .npy data is much smaller than the .nii scan, I can just store the entire dataset in GPU memory, which makes this run much faster
            volumes = torch.Tensor(np.array([np.load(path).squeeze() for path in npy_paths]))
            processed_data.append((volumes.cuda(), torch.Tensor(clin_vars).cuda(), dx))

        embedding_shape = tuple(processed_data[0][0].shape)
        all_volumes = torch.zeros((len(processed_data), *embedding_shape), device=torch.device('cuda'))
        all_clinvars = torch.zeros((len(processed_data), 3, len(settings.CLIN_VARS)), device=torch.device('cuda'))
        all_dxs = torch.zeros((len(processed_data)), device=torch.device('cuda')).type(torch.cuda.LongTensor)

        for i, (mat, cv, dx) in enumerate(processed_data):
            all_volumes[i, :] = mat
            all_clinvars[i, :] = cv
            all_dxs[i] = dx

        return LongitudinalDataset(all_volumes, all_clinvars, all_dxs)

    def generate_batch(self, idx_list):
        for i in range(0, len(idx_list), self.settings.BATCH_SIZE):
            batch_vols = self.vols[i : i + self.settings.BATCH_SIZE]
            batch_cvs = self.cvs[i : i + self.settings.BATCH_SIZE]
            batch_dxs = self.dxs[i : i + self.settings.BATCH_SIZE]

            yield batch_vols, batch_cvs, batch_dxs

    def full_training_set(self, idx = None):
        if idx is None:
            idx = self.train_idx

        return DataLoader(DatasetWrapper(idx, self.dataset), batch_size = self.settings.BATCH_SIZE, shuffle = False, num_workers = 0)

    def full_testing_set(self, idx = None):
        if idx is None:
            idx = self.test_idx

        return DataLoader(DatasetWrapper(idx, self.dataset), batch_size = self.settings.BATCH_SIZE, shuffle = False, num_workers = 0)

    def cross_validation_set(self, index_split = None):
        '''
        for train_index, val_index in self.skf.split(self.train_idx, self.train_label):
            train_set = DataLoader(DatasetWrapper(train_index, self.dataset), batch_size = self.settings.BATCH_SIZE, shuffle = False, num_workers = 0)
            val_set = DataLoader(DatasetWrapper(val_index, self.dataset), batch_size = self.settings.BATCH_SIZE, shuffle = False, num_workers = 0)
            yield train_set, val_set
        '''
        if self.settings.OUTER_SPLIT == 1:
            index_split = [i for i in range(len(self.train_idx))]

        if index_split is None and self.outer_skf is not None:
            for outer_train_index, test_index in self.outer_skf.split(self.train_idx, self.train_label):
                test_set = DataLoader(DatasetWrapper(self.train_idx[test_index], self.dataset), batch_size = self.settings.BATCH_SIZE, shuffle = False, num_workers = 0)
                yield outer_train_index, test_set
        else:
            for train_index, val_index in self.inner_skf.split(self.train_idx[index_split], self.train_label[index_split]):
                train_set = DataLoader(DatasetWrapper(self.train_idx[index_split][train_index], self.dataset), batch_size = self.settings.BATCH_SIZE, shuffle = False, num_workers = 0)
                val_set = DataLoader(DatasetWrapper(self.train_idx[index_split][val_index], self.dataset), batch_size = self.settings.BATCH_SIZE, shuffle = False, num_workers = 0)
                yield train_set, val_set
        








