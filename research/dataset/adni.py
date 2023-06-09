from collections import defaultdict
from typing import Tuple, List
import logging
import random
import os

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split

import research.common.dataset_config as dc
import research.dataset.adni_helper as helper
import research.dataset.util as util

IN_MEMORY = True


class _Dataset(Dataset):
    def __init__(self, cfg: dc.DatasetConfig, data_paths: List, device="cuda"):
        self.paths = []
        self.ni = []
        self.dxs = []
        self.num_samples = 0
        self.cpu = torch.device("cpu")
        self.device = torch.device(device)
        self.getter = None

        self.load_data(cfg, data_paths)

        if cfg.task == dc.DatasetTask.CLASSIFICATION:
            if cfg.mode == dc.DataMode.SCANS:
                if IN_MEMORY:
                    self.paths = [util.load_scan(path, self.cpu) for path in self.paths]
                    self.getter = self.get_scan_classification_memory
                else:
                    self.getter = self.get_scan_classification_disk
            elif cfg.mode == dc.DataMode.PATHS:
                self.getter = self.default_getter  # get_path_classification
        elif cfg.task == dc.DatasetTask.PREDICTION:
            if cfg.load_embeddings:
                self.paths = [
                    [
                        path.replace(cfg.scan_paths, cfg.embedding_paths).replace(
                            ".nii", ".npy"
                        )
                        for path in path_list
                    ]
                    for path_list in self.paths
                ]

            if cfg.mode == dc.DataMode.SCANS:
                if cfg.load_embeddings:
                    self.paths = self.load_embeddings(cfg)
                    self.getter = self.default_getter  # get_embedding_prediction
                else:
                    self.getter = self.get_scan_prediction
            elif cfg.mode == dc.DataMode.PATHS:
                self.paths = [util.join_paths(tp_paths) for tp_paths in self.paths]
                self.getter = self.default_getter  # get_path_prediction

        if self.getter is None:
            logging.error(f"Unknown task {cfg.task} and mode {cfg.mode}")

    def load_embeddings(self, cfg: dc.DatasetConfig):
        logging.info("Creating embeddings...")
        embeddings = []
        for embedding_paths in self.paths:
            embeddings.append([])
            for path in embedding_paths:
                if not os.path.exists(path):
                    logging.error(f"{path} does not exist")
                    exit(1)

                embeddings[-1].append(np.load(path).squeeze())

        embs = torch.tensor(np.array(embeddings), device=self.device)
        logging.info(f"Done ({embs.shape})")

        return embs

    def load_data(self, cfg: dc.DatasetConfig, data_paths: List):
        for ids, data, cohort in data_paths:
            ordinal = cohort.get_ordinal(cfg.cohorts)

            self.paths.append(ids)
            self.ni.append(data)
            self.dxs.append(ordinal)
            self.num_samples += 1

        if len(self.ni) > 0:
            self.ni = torch.tensor(
                np.array(self.ni), device=self.device, dtype=torch.float
            )

        self.dxs = torch.tensor(self.dxs, device=self.device, dtype=torch.long)

    def default_getter(self, path, ni, dx):
        return path, ni, dx

    def get_scan_classification_disk(self, path, ni, dx):
        return util.load_scan(path, self.device), ni, dx

    def get_scan_classification_memory(self, path, ni, dx):
        return path.to(self.device), ni, dx

    def get_scan_prediction(self, paths, ni, dx):
        paths = util.split_paths(paths)
        return (
            torch.stack([util.load_scan(path, self.device) for path in paths]),
            ni,
            dx,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.getter(self.paths[idx], self.ni[idx], self.dxs[idx])


class AdniDataset:
    def __init__(self, dataset_cfg: dc.DatasetConfig, **kwargs):
        self.cfg = dataset_cfg

        data_paths = helper.create_dataset(dataset_cfg)
        self.dataset = _Dataset(dataset_cfg, data_paths)

        self.idxs = torch.randperm(len(self.dataset))
        self.labels = self.dataset.dxs[self.idxs].cpu().numpy()

        split_type = self.cfg.split_type
        self.folds = []

        if isinstance(split_type, dc.NestedCV):
            logging.info(f"Nested CV: {len(self.idxs)} samples")

            outer_skf = StratifiedKFold(split_type.num_outer_fold)
            inner_skf = StratifiedKFold(split_type.num_inner_fold)

            for outer_idx, (full_train, test) in enumerate(
                outer_skf.split(self.idxs, self.labels)
            ):
                training_folds = []
                for inner_idx, (train, val) in enumerate(
                    inner_skf.split(full_train, self.labels[full_train])
                ):
                    logging.info(
                        f"Inner fold train set has {len(train)} samples, val set has {len(val)} samples"
                    )

                    train_loader = self.create_dataloader(full_train[train], proxy=True)
                    val_loader = self.create_dataloader(full_train[val], proxy=True)
                    training_folds.append((train_loader, val_loader))

                logging.info(
                    f"Full train set has {len(full_train)} samples, test set has {len(test)} samples"
                )

                full_train_loader = self.create_dataloader(full_train, proxy=True)
                test_loader = self.create_dataloader(test, proxy=True)

                self.folds.append((training_folds, full_train_loader, test_loader))

        elif isinstance(split_type, dc.FlatCV):
            logging.info(f"Flat CV: {len(self.idxs)} samples")
            skf = StratifiedKFold(split_type.num_folds)
            train_idxs, test_idxs, train_lab, test_lab = train_test_split(
                self.idxs, self.labels, test_size=split_type.test_ratio
            )

            training_folds = []
            for train, val in skf.split(train_idxs, train_lab):
                train_loader = self.create_dataloader(train_idxs[train])
                val_loader = self.create_dataloader(train_idxs[val])
                training_folds.append((train_loader, val_loader))

                logging.info(
                    f"Fold train set has {len(train)} samples, val set has {len(val)} samples"
                )

            logging.info(
                f"Full train set has {len(train_idxs)} samples, test set has {len(test_idxs)} samples"
            )
            full_train_loader = self.create_dataloader(train_idxs)
            test_loader = self.create_dataloader(test_idxs)
            self.folds.append((training_folds, full_train_loader, test_loader))
        elif isinstance(split_type, dc.BasicSplit):
            logging.info(f"Basic Split: {len(self.idxs)} samples")
            if split_type.sum() != 1:
                logging.error(
                    f"split_type.sum() must be 1, was {split_type.sum()} instead"
                )
                exit(1)

            full_train_idxs, test_idxs, train_labels, _ = train_test_split(
                self.idxs, self.labels, test_size=split_type.test_ratio
            )

            train_idxs, val_idxs, _, _ = train_test_split(
                full_train_idxs, train_labels, test_size=split_type.val_ratio
            )

            logging.info(
                f"Train set has {len(train_idxs)} samples, val has {len(val_idxs)} samples, test has {len(test_idxs)} samples"
            )
            train_loader = self.create_dataloader(train_idxs)
            val_loader = self.create_dataloader(val_idxs)
            test_loader = self.create_dataloader(test_idxs)
            self.folds.append((train_loader, val_loader, test_loader))
        elif isinstance(split_type, dc.NoSplit):
            logging.info(f"No Split: {len(self.idxs)} samples")
            self.folds = self.create_dataloader(self.idxs)
        else:
            logging.error(f"Unknown split type: {split_type}")
            exit(1)

    def create_dataloader(self, subset_idxs, proxy=False):
        if proxy:
            subset_idxs = self.idxs[subset_idxs]

        freq = defaultdict(int)
        for val in self.dataset.dxs[subset_idxs]:
            freq[val.item()] += 1

        logging.info(f"\t[{len(subset_idxs)}] {str(freq)}")
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            # shuffle=True,
            sampler=SubsetRandomSampler(subset_idxs),
        )

    def get_num_samples(self) -> int:
        return len(self.dataset)

    def get_batch_size(self) -> int:
        return self.cfg.batch_size

    def get_split_type(self) -> dc.SplitTypes:
        return self.cfg.split_type

    def get_data_shape(self) -> Tuple[Tuple[int], ...]:
        scan, ni, dx = self.dataset[0]
        if (
            self.cfg.task == dc.DatasetTask.PREDICTION
            and self.cfg.mode == dc.DataMode.PATHS
        ):
            if self.cfg.load_embeddings:
                scan = util.split_paths(scan)
                scan = np.array([np.load(path).squeeze() for path in scan])
                scan = torch.tensor(scan, device=self.dataset.device, dtype=torch.float)
            else:
                scan, ni, dx = self.dataset.get_scan_prediction(scan, ni, dx)
        elif (
            self.cfg.task == dc.DatasetTask.CLASSIFICATION
            and self.cfg.mode == dc.DataMode.PATHS
        ):
            scan, ni, dx = self.dataset.get_scan_classification_disk(scan, ni, dx)

        if self.cfg.task == dc.DatasetTask.CLASSIFICATION:
            num_out = 3 if len(self.cfg.cohorts) == 0 else len(self.cfg.cohorts)
        elif self.cfg.task == dc.DatasetTask.PREDICTION:
            num_out = 2

        return scan.shape, ni.shape, (num_out,)

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
