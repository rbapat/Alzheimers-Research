import logging
from typing import Tuple, List
import random
import os

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split

import research.common.dataset_config as dc
import research.dataset.adni_helper as helper


class _Dataset(Dataset):
    def __init__(self, cfg: dc.DatasetConfig, data_paths: List, device="cuda"):
        self.paths = []
        self.ni = []
        self.dxs = []
        self.num_samples = 0
        self.device = torch.Device(device)
        self.getter = None

        self.load_data(cfg, data_paths)

        if cfg.task == dc.DatasetTask.CLASSIFICATION:
            if cfg.mode == dc.DataMode.SCANS:
                self.getter = self.get_scan_classification
            elif cfg.mode == dc.DataMode.PATHS:
                self.getter = self.default_getter  # get_path_classification
        elif cfg.task == dc.DatasetTask.PREDICTION:
            if cfg.load_embeddings:
                self.paths = [
                    [
                        [
                            path.replace(cfg.scan_paths, cfg.embedding_paths).replace(
                                ".nii", ".npy"
                            )
                            for path in path_list
                        ]
                    ]
                    for path_list in self.paths
                ]

            if cfg.mode == dc.DataMode.SCANS:
                if cfg.load_embeddings:
                    self.paths = self.load_embeddings()
                    self.getter = self.default_getter  # get_embedding_prediction
                else:
                    self.getter = self.get_scan_prediction
            elif cfg.mode == dc.DataMode.PATHS:
                self.getter = self.default_getter  # get_path_prediction

        if self.getter is None:
            logging.error(f"Unknown task {cfg.task} and mode {cfg.mode}")

    def load_embeddings(self, cfg: dc.DatasetConfig):
        embeddings = []
        for embedding_paths in self.paths:
            embeddings.append([])
            for path in embedding_paths:
                if not os.path.exists(path):
                    logging.error(f"{path} does not exist")
                    exit(1)

                embeddings[-1].append(np.load(path).squeeze())

        return torch.tensor(np.array(embeddings), device=self.device)

    def load_data(self, cfg: dc.DatasetConfig, data_paths: List):
        for ids, data, cohort in data_paths:
            self.paths.append(ids)
            self.ni.append(data)
            self.dxs.append(cohort.get_ordinal(cfg.cohorts))
            self.num_samples += 1

        if self.ni[0] is not None:
            self.ni = torch.Tensor(np.array(self.ni), device=self.device)

        self.dxs = torch.LongTensor(self.dxs, device=self.device)

    def default_getter(self, path, ni, dx):
        return path, ni, dx

    def get_scan_classification(self, path, ni, dx):
        mat = nib.load(path).get_fdata()
        mat = (mat - mat.min()) / (mat.max() - mat.min())  # min-max normalization
        return torch.Tensor(mat, device=self.device), ni, dx

    def get_scan_prediction(self, paths, ni, dx):
        mats = []
        for path in paths:
            mat = nib.load(path).get_fdata()
            mat = (mat - mat.min()) / (mat.max() - mat.min())  # min-max normalization
            mats.append(torch.Tensor(mat, device=self.device))

        return torch.cat(mats), ni, dx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.getter(self.paths[idx], self.ni[idx], self.dxs[idx])


class AdniDataset:
    def __init__(self, dataset_cfg: dc.DatasetConfig, **kwargs):
        self.cfg = dataset_cfg

        data_paths = helper.create_dataset(dataset_cfg)
        self.dataset = _Dataset(dataset_cfg, data_paths)

        self.idxs = list(range(len(self.dataset)))
        random.shuffle(self.idxs)
        self.labels = self.dataset.dxs[self.idxs].cpu().numpy()

        split_type = self.cfg.split_type
        self.folds = []

        if isinstance(split_type, dc.NestedCV):
            outer_skf = StratifiedKFold(split_type.num_outer_fold)
            inner_skf = StratifiedKFold(split_type.num_inner_fold)

            for full_train, test in outer_skf.split(self.idxs, self.labels):
                training_folds = []
                for train, val in inner_skf.split(full_train, self.labels[full_train]):
                    train_loader = self.create_dataloader(full_train[train], proxy=True)
                    val_loader = self.create_dataloader(full_train[val], proxy=True)
                    training_folds.append((train_loader, val_loader))

                full_train_loader = self.create_dataloader(full_train, proxy=True)
                test_loader = self.create_dataloader(test)
                self.folds.append((training_folds, full_train_loader, test_loader))

        elif isinstance(split_type, dc.FlatCV):
            skf = StratifiedKFold(split_type.num_folds)
            train_idxs, test_idxs, train_lab, test_lab = train_test_split(
                self.idxs, self.labels, test_size=split_type.test_ratio
            )

            training_folds = []
            for train, val in skf.split(train_idxs, train_lab):
                train_loader = self.create_dataloader(train_idxs[train])
                val_loader = self.create_dataloader(train_idxs[val])
                training_folds.append((train_loader, val_loader))

            full_train_loader = self.create_dataloader(train_idxs)
            test_loader = self.create_dataloader(test_idxs)
            self.folds.append((training_folds, full_train_loader, test_loader))
        elif isinstance(split_type, dc.BasicSplit):
            assert split_type.sum() == 1

            full_train_idxs, test_idxs, train_labels, _ = train_test_split(
                self.idxs, self.labels, test_size=split_type.test_ratio
            )

            train_idxs, val_idxs, _, _ = train_test_split(
                full_train_idxs, train_labels, test_size=split_type.val_ratio
            )

            train_loader = self.create_dataloader(train_idxs)
            val_loader = self.create_dataloader(val_idxs)
            test_loader = self.create_dataloader(test_idxs)
            self.folds.append((train_loader, val_loader, test_loader))

        else:
            logging.error(f"Unknown split type: {split_type}")
            exit(1)

    def create_dataloader(self, subset_idxs, proxy=False):
        if proxy:
            subset_idxs = self.idxs[subset_idxs]

        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            sampler=SubsetRandomSampler(subset_idxs),
        )

    def get_split_type(self) -> dc.SplitTypes:
        return self.cfg.split_type

    def get_data_shape(self) -> Tuple[Tuple[int], ...]:
        if self.cfg.mode == dc.DataMode.SCANS or (
            self.cfg.task == dc.DatasetTask.PREDICTION and not self.cfg.load_embeddings
        ):
            logging.error(
                "You must be in SCANS mode and load embeddings in order to get the data shape"
            )
            exit(1)

        scans, ni, dx = self.dataset[0]
        if self.cfg.task == dc.DatasetTask.CLASSIFICATION:
            num_out = 3 if self.cfg.cohorts is None else len(self.cfg.cohorts)
        elif self.cfg.task == dc.DatasetTask.PREDICTION:
            num_out = 2

        return scans.shape, ni.shape, (num_out,)

    def get_data(self):
        """Returns the training, validation, and testing data for this dataset configuration (for one epoch)

        - If `isinstance(self.cfg.split_type, NestedCV)`, then this returns:
            - a list with a tuple of dataloaders
            - each of these tuples contain (training_data, full_train_loader, test_dataloader)
            - training_data is a list of tuple with (training_dataloader, val_dataloader)
        - If `isinstance(self.cfg.split_type, FlatCV)`, then this returns:
            - a tuple of (training_data, test_dataloader)
            - training_data is a list of tuple with (training_dataloader, full_train_loader, val_dataloader)
        - If `isinstance(self.cfg.split_type, BasicSplit)`, then this returns:
            - a tuple with (training_dataloader, val_dataloader, test_dataloader)
        """
        return self.folds
