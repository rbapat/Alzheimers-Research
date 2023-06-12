import sys
import logging
from typing import Tuple

import torch
import torch.nn as nn
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F

from research.tasks.logger import Logger
import research.common.dataset_config as dc
from research.dataset.adni import AdniDataset
from research.common.abstract import AbstractTask
from research.common.config_classes import TrainConfig


class TrainTask(AbstractTask):
    def __init__(
        self, dataset: AdniDataset, train_cfg: TrainConfig, logger: Logger, **kwargs
    ):
        self.dataset = dataset
        self.train_cfg = train_cfg

        self.logger = logger

        mri_shape, ni_shape, out_shape = self.dataset.get_data_shape()

        model_args = kwargs["model"]
        self.model_args = {} if model_args is None else model_args
        self.model_args["mri_shape"] = mri_shape
        self.model_args["ni_shape"] = ni_shape
        self.model_args["out_shape"] = out_shape

        logging.info("Train task initialized")

    def init_model(self) -> Tuple[nn.Module, optim.Optimizer, nn.Module]:
        model = self.train_cfg.model_cls(**self.model_args).cuda()
        optim = self.train_cfg.optim(model.parameters())
        criterion = self.train_cfg.loss_function

        if self.train_cfg.model_weights is not None:
            model.load_state_dict(torch.load(self.train_cfg.model_weights))

        return model, optim, criterion

    def evaluate_model(
        self, train_set, val_set, save_weights: bool, additional: Tuple = ()
    ) -> torch.Tensor:
        """Trains the model using the optimizer and loss function in the training config on the given
        training set and validation set.

        Args:
            train_set (DataLoader): dataset to train on
            val_set (DataLoader): dataset to evaluate
            save_weights (bool): if we should save the weights after each epoch
            additional (Tuple[DataLoader]): additional dataloaders that we should evaluate

        Returns:
            torch.Tensor: training results with shape `(num_phases, num_epochs, 4)`. For each phase (training, validation, etc.),
            this will track the average loss, balanced accuracy, sensitivity, and specificity.
        """

        phases = (train_set, val_set, *additional)
        names = ["Train", "Validation"]
        for _ in range(len(phases) - 2):
            names.append("Add 1")

        total_epochs = self.train_cfg.num_epochs
        model, optimizer, criterion = self.init_model()
        es_tolerance = self.train_cfg.es_tolerance

        results = torch.zeros(len(phases), self.train_cfg.num_epochs, 4)
        best_loss, tolerance_track = sys.float_info.max, 0
        for epoch in range(total_epochs):
            if es_tolerance != 0 and tolerance_track > es_tolerance:
                break

            losses = []
            for phase, loader in enumerate(phases):
                preds = []
                corrects = []
                logs = []
                ptids = []
                total_loss = 0
                num_in_epoch = 0

                for mat, clin_vars, ground_truth, ptid in loader:
                    optimizer.zero_grad()
                    model.train(phase == 0)

                    raw_output = model(mat, clin_vars)
                    predictions = torch.argmax(F.softmax(raw_output, dim=1), dim=1)
                    loss = criterion(raw_output, ground_truth)

                    total_loss += loss.item() * len(mat)
                    num_in_epoch += len(mat)
                    for gt, p, logits, bptid in zip(
                        ground_truth, predictions, raw_output, ptid
                    ):
                        corrects.append(gt.item())
                        preds.append(p.item())
                        logs.append(logits.detach().cpu().numpy())
                        ptids.append(bptid)

                    if phase == 0:
                        loss.backward()
                        optimizer.step()

                entry = results[phase, epoch]
                entry[0] = total_loss / num_in_epoch
                entry[1] = metrics.balanced_accuracy_score(corrects, preds)
                entry[2] = metrics.recall_score(corrects, preds, pos_label=1)
                entry[3] = metrics.recall_score(corrects, preds, pos_label=0)
                losses.append(entry[0].item())

                # if phase == 1:
                #     wrongs = {}
                #     for c, p, l, pt in zip(corrects, preds, logs, ptids):
                #         if c != p:
                #             msg = f"Predicted {p}, Actual {c}, Logits {[round(li.item(), 4) for li in l]} | {pt}"
                #             wrongs[pt] = msg

                #     for idx, key in enumerate(dict(sorted(wrongs.items()))):
                #         print(f"[{idx}] {wrongs[key]}")
                #     input()

            self.logger.epoch_new(
                epoch + 1,
                total_epochs,
                results[:, epoch, :],
                names,
                model if save_weights else None,
            )

            if losses[1] < best_loss:
                best_loss = losses[1]
                tolerance_track = 0
            else:
                tolerance_track += 1

        return results

    def nested_cv(self, split_type: dc.NestedCV):
        logging.info("Doing nested CV...")
        train_results = torch.zeros(
            split_type.num_outer_fold,
            split_type.num_inner_fold,
            2,
            self.train_cfg.num_epochs,
            4,
        )

        # from sklearn.decomposition import PCA
        # import matplotlib.pyplot as plt
        # import numpy as np

        # embs = self.dataset.dataset.paths[idxs].cpu().numpy()  # (num_test, 3, emb_size)

        # for tp in range(3):
        #     dxs = self.dataset.dataset.dxs[idxs].cpu().numpy()  # (num_test, 1)
        #     ptids = np.array([self.dataset.dataset.ptids[i] for i in idxs])

        #     pca = PCA(n_components=2)
        #     embs_pca = embs[:, tp, :]
        #     embs_pca = pca.fit(embs_pca).transform(embs_pca)

        #     plt.title(f"Timepoint {tp}")
        #     for color, i, target_name in zip(["g", "r"], [0, 1], ["sMCI", "pMCI"]):
        #         X = embs_pca[dxs == i, 0]
        #         Y = embs_pca[dxs == i, 1]
        #         P = ptids[dxs == i]

        #         plt.scatter(
        #             X,
        #             Y,
        #             color=color,
        #             alpha=0.8,
        #             lw=2,
        #             label=target_name,
        #         )

        #         # for xval, yval, label in zip(X, Y, P):
        #         #     plt.annotate(label, xy=(xval, yval))
        #     plt.figure()

        # plt.show()
        # exit(1)

        test_results = torch.zeros(
            split_type.num_outer_fold, 2, self.train_cfg.num_epochs, 4
        )

        for outer_idx, (inner_fold, full_train_loader, test_loader) in enumerate(
            self.dataset.get_data()
        ):
            logging.info(f"Outer fold {outer_idx+1}/{split_type.num_outer_fold}")
            for inner_idx, (train_loader, val_loader) in enumerate(inner_fold):
                logging.info(f"Inner fold {inner_idx+1}/{split_type.num_inner_fold}")
                train_results[outer_idx, inner_idx, :] = self.evaluate_model(
                    train_loader, val_loader, False
                )

            logging.info(f"Evaluating outer fold {outer_idx+1}")
            test_results[outer_idx, :] = self.evaluate_model(
                full_train_loader, test_loader, True
            )

        self.logger.save_results(train_results, "train")
        self.logger.save_results(test_results, "test")
        logging.info("Done!")

    def flat_cv(self, split_type: dc.FlatCV):
        logging.info("Doing flat CV...")
        train_results = torch.zeros(
            split_type.num_folds, 2, self.train_cfg.num_epochs, 4
        )

        folds, test_loader = self.dataset.get_data()
        for fold_idx, (train_loader, full_train_loader, val_loader) in enumerate(folds):
            train_results[fold_idx, :] = self.evaluate_model(
                train_loader, val_loader, False
            )

        test_results = self.evaluate_model(full_train_loader, test_loader, True)
        self.logger.save_results(train_results, "train")
        self.logger.save_results(test_results, "test")
        logging.info("Done!")

    def basic_split(self, split_type: dc.BasicSplit):
        logging.info("Doing a basic split...")

        # from sklearn.decomposition import PCA
        # import matplotlib.pyplot as plt

        # ni = self.dataset.dataset.ni.cpu().numpy()  # [num_samples, num_ni_vars]
        # pca = PCA(n_components=2)
        # ni_pca = pca.fit(ni).transform(ni)

        # dxs = self.dataset.dataset.dxs.cpu().numpy()
        # for color, i, target_name in zip(["g", "r"], [0, 1], ["sMCI", "pMCI"]):
        #     plt.scatter(
        #         ni_pca[dxs == i, 0],
        #         ni_pca[dxs == i, 1],
        #         color=color,
        #         alpha=0.8,
        #         lw=2,
        #         label=target_name,
        #     )

        # plt.show()
        # exit(1)
        train_loader, val_loader, test_loader = self.dataset.get_data()[0]
        results = self.evaluate_model(
            train_loader, val_loader, True, additional=(test_loader,)
        )

        self.logger.save_results(results, "train_test")
        logging.info("Done!")

    def run(self):
        logging.info("Starting training task...")
        split_type = self.dataset.get_split_type()
        if isinstance(split_type, dc.NestedCV):
            self.nested_cv(split_type)
        elif isinstance(split_type, dc.FlatCV):
            self.flat_cv(split_type)
        elif isinstance(split_type, dc.BasicSplit):
            self.basic_split(split_type)
        else:
            logging.error(f"Unknown split type: {split_type}")
            exit(1)
