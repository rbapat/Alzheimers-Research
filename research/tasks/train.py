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

        results = torch.zeros(len(phases), self.train_cfg.num_epochs, 4)
        for epoch in range(total_epochs):
            for phase, loader in enumerate(phases):
                preds = []
                corrects = []
                total_loss = 0
                num_in_epoch = 0

                for mat, clin_vars, ground_truth in loader:
                    optimizer.zero_grad()
                    model.train(phase == 0)

                    raw_output = model(mat, clin_vars)
                    predictions = torch.argmax(F.softmax(raw_output, dim=1), dim=1)
                    loss = criterion(raw_output, ground_truth)

                    total_loss += loss.item() * len(mat)
                    num_in_epoch += len(mat)
                    for gt, p in zip(ground_truth, predictions):
                        corrects.append(gt.item())
                        preds.append(p.item())

                    if phase == 0:
                        loss.backward()
                        optimizer.step()

                # if phase == 1:
                #     logging.info(preds)
                #     logging.info(corrects)
                entry = results[phase, epoch]
                entry[0] = total_loss / num_in_epoch
                entry[1] = metrics.balanced_accuracy_score(corrects, preds)
                entry[2] = metrics.recall_score(corrects, preds, pos_label=1)
                entry[3] = metrics.recall_score(corrects, preds, pos_label=0)

            self.logger.epoch_new(
                epoch + 1,
                total_epochs,
                results[:, epoch, :],
                names,
                model if save_weights else None,
            )
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

        # import pandas as pd

        # subs = ["PTID", "IMAGEUID", "DX", "Month"]
        # df = pd.read_csv("ADNIMERGE.csv", low_memory=False).dropna(subset=subs)[subs]

        # print("TEST")
        # for num, tidx in enumerate(test_idx):
        #     ptid, dx = self.dataset.dataset.ptids[tidx], self.dataset.dataset.dxs[tidx]

        #     print(f"{num}: {'STABLE' if dx.item() == 0 else 'PROGRESSIVE'}")
        #     print((df[df["PTID"] == ptid]).sort_values(by=["Month"]))
        #     input()
        # print("TRAIN")
        # for num, tidx in enumerate(train_idx):
        #     ptid, dx = self.dataset.dataset.ptids[tidx], self.dataset.dataset.dxs[tidx]

        #     print(f"{num}: {'STABLE' if dx.item() == 0 else 'PROGRESSIVE'}")
        #     print((df[df["PTID"] == ptid]).sort_values(by=["Month"]))
        #     input()

        # input()

        # from sklearn.decomposition import PCA
        # import matplotlib.pyplot as plt
        # import numpy as np

        # train_idx = [
        #     161,
        #     51,
        #     136,
        #     160,
        #     35,
        #     101,
        #     57,
        #     63,
        #     71,
        #     129,
        #     30,
        #     150,
        #     88,
        #     105,
        #     133,
        #     154,
        #     34,
        #     98,
        #     64,
        #     125,
        #     146,
        #     122,
        #     168,
        #     126,
        #     75,
        #     58,
        #     21,
        #     144,
        #     153,
        #     143,
        #     156,
        #     9,
        #     139,
        #     48,
        #     83,
        #     110,
        #     115,
        #     12,
        #     132,
        #     120,
        #     73,
        #     74,
        #     6,
        #     81,
        #     107,
        #     8,
        #     18,
        #     171,
        #     113,
        #     140,
        #     106,
        #     1,
        #     49,
        #     72,
        #     69,
        #     170,
        #     163,
        #     20,
        #     67,
        #     102,
        #     15,
        #     91,
        #     23,
        #     92,
        #     84,
        #     123,
        #     28,
        #     152,
        #     7,
        #     157,
        #     100,
        #     4,
        #     85,
        #     70,
        #     55,
        #     50,
        #     13,
        #     135,
        #     142,
        #     77,
        #     45,
        #     39,
        #     24,
        #     111,
        #     22,
        #     127,
        #     87,
        #     164,
        #     141,
        #     26,
        #     114,
        #     95,
        #     151,
        #     124,
        #     10,
        #     86,
        #     96,
        #     38,
        #     3,
        #     44,
        #     162,
        #     134,
        #     148,
        #     46,
        #     155,
        #     89,
        #     41,
        #     103,
        #     82,
        #     59,
        #     108,
        #     29,
        #     37,
        #     16,
        #     14,
        #     93,
        #     119,
        #     53,
        #     94,
        #     112,
        #     158,
        #     68,
        #     60,
        #     56,
        #     11,
        #     104,
        #     167,
        #     79,
        #     54,
        #     159,
        #     65,
        #     169,
        #     76,
        #     138,
        #     118,
        #     31,
        #     42,
        #     145,
        # ]
        # test_idx = [
        #     52,
        #     36,
        #     121,
        #     33,
        #     27,
        #     2,
        #     47,
        #     137,
        #     117,
        #     25,
        #     62,
        #     0,
        #     40,
        #     78,
        #     147,
        #     109,
        #     97,
        #     66,
        #     17,
        #     131,
        #     149,
        #     165,
        #     32,
        #     99,
        #     130,
        #     61,
        #     116,
        #     5,
        #     128,
        #     19,
        #     80,
        #     43,
        #     90,
        #     166,
        # ]

        # target = test_idx
        # ni = (
        #     self.dataset.dataset.ni[target].view(len(target), -1).cpu().numpy()
        # )  # [num_samples, 3, num_ni_vars]
        # pca = PCA(n_components=2)
        # ni_pca = pca.fit(ni).transform(ni)

        # dxs = self.dataset.dataset.dxs[target].cpu().numpy()
        # ptid = np.array(self.dataset.dataset.ptids)[target]
        # for color, i, target_name in zip(["g", "r"], [0, 1], ["sMCI", "pMCI"]):
        #     X = ni_pca[dxs == i, 0]
        #     Y = ni_pca[dxs == i, 1]
        #     T = ptid[dxs == i]
        #     plt.scatter(
        #         X,
        #         Y,
        #         color=color,
        #         alpha=0.8,
        #         lw=2,
        #         label=target_name,
        #     )

        #     for i, txt in enumerate(T):
        #         plt.annotate(txt, (X[i], Y[i]))
        #         print(txt)

        # plt.show()
        # exit(1)

        test_results = torch.zeros(
            split_type.num_outer_fold, 2, self.train_cfg.num_epochs, 4
        )
        # targets = [4, -1]
        for outer_idx, (inner_fold, full_train_loader, test_loader) in enumerate(
            self.dataset.get_data()
        ):
            logging.info(f"Outer fold {outer_idx+1}/{split_type.num_outer_fold}")
            for inner_idx, (train_loader, val_loader) in enumerate(inner_fold):
                # if inner_idx != targets[1]:
                #     continue
                logging.info(f"Inner fold {inner_idx+1}/{split_type.num_inner_fold}")
                train_results[outer_idx, inner_idx, :] = self.evaluate_model(
                    train_loader, val_loader, False
                )

            # if outer_idx != targets[0]:
            #     continue

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
