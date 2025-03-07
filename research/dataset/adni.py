from collections import defaultdict
from typing import Tuple, List
import logging
import random
import os

import torch
import numpy as np
import torch.nn.functional as F
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
        self.ptids = []
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
        ni_data = []
        for ptid, ids, data, cohort in data_paths:
            ordinal = cohort.get_ordinal(cfg.cohorts)

            self.ptids.append(ptid)
            self.paths.append(ids)
            ni_data.append(data)
            self.dxs.append(ordinal)
            self.num_samples += 1

        if ni_data[0] is not None:
            ni_data = torch.tensor(np.array(ni_data), dtype=torch.float)
            num_samples, num_tp, num_ni = ni_data.shape
            ni_data = ni_data.reshape(num_samples * num_tp, num_ni)

            cat_ni = []
            for idx, ni_dict in enumerate(cfg.ni_vars):
                data = ni_data[:, idx]
                if ni_dict["type"] == "continuous_bounded":
                    mins = data[data < ni_dict["min"]]
                    maxes = data[data > ni_dict["max"]]
                    if len(mins) > 0:
                        logging.info(
                            f"Found a {ni_dict['type']} value under {ni_dict['min']}: {mins}"
                        )
                        exit(1)
                    elif len(maxes) > 0:
                        logging.info(
                            f"Found a {ni_dict['type']} value above {ni_dict['max']}: {maxes}"
                        )
                        exit(1)
                    elif ni_dict["max"] == 0:
                        logging.info(f"Max value for {ni_dict['type']} cannot be 0")
                        exit(1)

                    data = (data - ni_dict["min"]) / ni_dict["max"]
                    data = data.unsqueeze(-1)

                elif ni_dict["type"] == "normal":
                    mean, std = torch.mean(data), torch.std(data)
                    if std == 0:
                        logging.info(f"Std Dev for {ni_dict['type']} cannot be 0")
                        exit(1)

                    data = (data - mean) / std
                    data = data.unsqueeze(-1)
                elif ni_dict["type"] == "discrete":
                    data = data.to(torch.long)
                    data = F.one_hot(data, num_classes=ni_dict["num_classes"])
                else:
                    logging.error(f"Unknown ni_dict type: {ni_dict['type']}")
                    exit(1)

                cat_ni.append(data)

            ni_data = torch.cat(cat_ni, dim=-1).view(num_samples, num_tp, -1)
            self.ni = ni_data.to(self.device)

            # import matplotlib.pyplot as plt

            # for idx, ni_name in enumerate(cfg.ni_vars):
            #     plt.title(ni_name)
            #     plt.hist(
            #         ni[:, idx],
            #         bins=np.arange(ni[:, idx].min(), ni[:, idx].max() + 1),
            #         align="left",
            #     )
            #     plt.figure()
            # plt.show()

        self.dxs = torch.tensor(self.dxs, device=self.device, dtype=torch.long)

    def default_getter(self, path, ni, dx, ptid):
        return path, ni, dx, ptid

    def get_scan_classification_disk(self, path, ni, dx, ptid):
        return util.load_scan(path, self.device), ni, dx, ptid

    def get_scan_classification_memory(self, path, ni, dx, ptid):
        return path.to(self.device), ni, dx, ptid

    def get_scan_prediction(self, paths, ni, dx, ptid):
        paths = util.split_paths(paths)
        return (
            torch.stack([util.load_scan(path, self.device) for path in paths]),
            ni,
            dx,
            ptid,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.getter(
            self.paths[idx], self.ni[idx], self.dxs[idx], self.ptids[idx]
        )


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
                train_loader = self.create_dataloader(train_idxs[train], proxy=True)
                val_loader = self.create_dataloader(train_idxs[val], proxy=True)
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
                self.idxs,
                self.labels,
                train_size=split_type.train_ratio + split_type.val_ratio,
            )

            if split_type.val_ratio != 0:
                train_idxs, val_idxs, _, _ = train_test_split(
                    full_train_idxs, train_labels, test_size=split_type.val_ratio
                )
            else:
                train_idxs = full_train_idxs
                val_idxs = []

            logging.info(
                f"Train set has {len(train_idxs)} samples, val has {len(val_idxs)} samples, test has {len(test_idxs)} samples"
            )
            train_loader = self.create_dataloader(train_idxs)
            val_loader = (
                None if len(val_idxs) == 0 else self.create_dataloader(val_idxs)
            )
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
        # logging.info(f"\t\t{subset_idxs}")
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
        scan, ni, dx, ptid = self.dataset[0]
        if (
            self.cfg.task == dc.DatasetTask.PREDICTION
            and self.cfg.mode == dc.DataMode.PATHS
        ):
            if self.cfg.load_embeddings:
                scan = util.split_paths(scan)
                scan = np.array([np.load(path).squeeze() for path in scan])
                scan = torch.tensor(scan, device=self.dataset.device, dtype=torch.float)
            else:
                scan, ni, dx, ptid = self.dataset.get_scan_prediction(
                    scan, ni, dx, ptid
                )
        elif (
            self.cfg.task == dc.DatasetTask.CLASSIFICATION
            and self.cfg.mode == dc.DataMode.PATHS
        ):
            scan, ni, dx, ptid = self.dataset.get_scan_classification_disk(
                scan, ni, dx, ptid
            )

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
