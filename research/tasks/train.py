from research.common.abstract import AbstractTask
from research.dataset.adni import AdniDataset
import research.common.dataset_config as dc
import logging

import torch.nn as nn


class TrainTask(AbstractTask):
    def __init__(self, dataset: AdniDataset, model_cls: type, **kwargs):
        self.dataset = dataset
        self.model_cls = model_cls

    def get_model(self) -> nn.Module:
        mri_shape, ni_shape, out_shape = self.dataset.get_data_shape()
        return self.model_cls(mri_shape, ni_shape, out_shape)

    def nested_cv(self):
        self.dataset.get_data
        for inner_fold, test_loader in self.dataset.get_data():
            for train_loader, val_loader in inner_fold:
                pass  # TODO: train and track (train, val)
            pass  # TODO: train and track (all train, val)

    def flat_cv(self):
        folds, test_loader = self.dataset.get_data()
        for train_loader, val_loader in folds:
            pass  # TODO: train and track (train, val)

        # TODO: train and track (all train, test)

    def basic_split(self):
        train_loader, val_loader, test_loader = self.dataset.get_data()
        # TODO: train and track (train, val)
        # TODO: train and track (all train, test)

    def run(self):
        split_type = self.dataset.get_split_type()
        if isinstance(split_type, dc.NestedCV):
            self.nested_cv()
        elif isinstance(split_type, dc.FlatCV):
            self.flat_cv()
        elif isinstance(split_type, dc.BasicSplit):
            self.basic_split()
        else:
            logging.error(f"Unknown split type: {split_type}")
            exit(1)
