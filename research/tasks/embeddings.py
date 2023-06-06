import logging
from typing import Tuple, List

import os
import torch
import numpy as np


import research.dataset.util as util
from research.tasks.logger import Logger
from research.dataset.adni import AdniDataset
from research.common.abstract import AbstractTask
from research.common.config_classes import EmbeddingConfig


class EmbeddingTask(AbstractTask):
    def __init__(
        self,
        dataset: AdniDataset,
        embedding_cfg: EmbeddingConfig,
        logger: Logger,
        **kwargs,
    ):
        self.dataset = dataset
        self.embedding_cfg = embedding_cfg
        self.logger = logger
        self.device = torch.device("cuda")

        mri_shape, ni_shape, out_shape = self.dataset.get_data_shape()

        # we know the shape is (3, 182, 218, 182) but we want to process one at a time
        mri_shape = mri_shape[1:]

        model_args = kwargs["model"]
        self.model_args = {} if model_args is None else model_args
        self.model_args["mri_shape"] = mri_shape
        self.model_args["ni_shape"] = ni_shape
        self.model_args["out_shape"] = out_shape

        if os.path.exists(embedding_cfg.embedding_path):
            logging.error(
                "Embedding path already exists, I don't want to overwrite any existing embeddings."
            )
            exit(1)

        if not os.path.exists(embedding_cfg.scan_path):
            logging.error(f"Scan path {embedding_cfg.scan_path} does not exist")
            exit(1)

        if not os.path.exists(embedding_cfg.weight_path):
            logging.error(f"Weight path {embedding_cfg.weight_path} does not exist")
            exit(1)

        logging.info("Embedding task initialized")

    def run(self):
        model = self.embedding_cfg.model_cls(**self.model_args).cuda()
        self.logger.load_weights(model, self.embedding_cfg.weight_path)

        loader = self.dataset.get_data()
        logging.info(
            f"Creating embeddings for {self.dataset.get_num_samples()} files..."
        )

        num_samples = self.dataset.get_num_samples()
        batch_size = self.dataset.get_batch_size()

        for batch_num, (batch, ni, dx) in enumerate(loader):
            for i, tp_paths in enumerate(batch):
                logging.info(f"[{batch_num*batch_size+i+1}/{num_samples}]")
                for old_path in util.split_paths(tp_paths):
                    new_path = old_path.replace(
                        self.embedding_cfg.scan_path, self.embedding_cfg.embedding_path
                    ).replace(".nii", ".npy")

                    mat = util.load_scan(old_path, self.device)

                    features = model.features(mat).cpu().detach().numpy().squeeze()

                    os.makedirs(os.path.dirname(new_path))
                    np.save(new_path, features)
                    logging.info(f"\tSaved {new_path}")

        logging.info("Done!")
