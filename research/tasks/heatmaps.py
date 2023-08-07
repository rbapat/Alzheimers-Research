import logging
from typing import Tuple, List

import torch
import numpy as np
import torch.nn as nn
import skimage.transform
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from research.tasks.logger import Logger
import research.common.dataset_config as dc
import research.dataset.util as util
from research.dataset.adni import AdniDataset
from research.common.abstract import AbstractTask
from research.common.config_classes import HeatmapsConfig


plt.style.use("dark_background")
COLORMAP = plt.cm.get_cmap("YlOrRd").copy()
COLORMAP.set_under("k", alpha=0)


def get_children(model):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


class GradCAMNet(nn.Module):
    def __init__(
        self,
        heatmap_cfg: HeatmapsConfig,
        dataset: AdniDataset,
        embedding_params,
        prediction_params,
        **kwargs,
    ):
        super().__init__()

        mri_shape, ni_shape, out_shape = dataset.get_data_shape()
        num_tp = len(mri_shape)

        embedding_params = {} if embedding_params is None else embedding_params
        embedding_params["mri_shape"] = mri_shape[1:]
        embedding_params["ni_shape"] = ni_shape[1:]
        embedding_params["out_shape"] = out_shape
        self.embedding_model = heatmap_cfg.embedding_model_cls(**embedding_params)

        prediction_params = {} if prediction_params is None else prediction_params
        prediction_params["mri_shape"] = (
            num_tp,
            *self.embedding_model.get_features_shape(),
        )
        prediction_params["ni_shape"] = ni_shape
        prediction_params["out_shape"] = out_shape
        self.prediction_model = heatmap_cfg.prediction_model_cls(**prediction_params)

        self.activation_maps = []
        self.image_reconstruction = []

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop()
            # for the forward pass, after the ReLU operation,
            # if the output value is positive, we set the value to 1,
            # and if the output value is negative, we set it to 0.
            grad[grad > 0] = 1

            # grad_out[0] stores the gradients for each feature map,
            # and we only retain the positive gradients
            positive_grad_out = torch.clamp(grad_in[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)

        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction.append(grad_in[0].clone())

        self.embedding_model.stem.register_full_backward_hook(first_layer_hook_fn)
        for module in get_children(self):
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_full_backward_hook(backward_hook_fn)

    def forward(self, x, clin_vars):
        bs, seq_len, d1, d2, d3 = x.shape

        x = x.view(bs * seq_len, d1, d2, d3)

        x = self.embedding_model.features(x)

        x = x.view(bs, seq_len, -1)

        x = self.prediction_model(x, clin_vars)
        return x


class HeatmapsTask(AbstractTask):
    def __init__(
        self,
        heatmaps_cfg: HeatmapsConfig,
        dataset: AdniDataset,
        logger: Logger,
        embedding_params: dict,
        prediction_params: dict,
        **kwargs,
    ):
        self.dataset = dataset
        self.cfg = heatmaps_cfg
        self.logger = logger
        self.device = torch.device("cuda")

        self.embedding_params = embedding_params
        self.prediction_params = prediction_params

        logging.info("Heatmaps task initialized")

    def get_model(self) -> Tuple[GradCAMNet, nn.Module]:
        cam_net = GradCAMNet(
            self.cfg, self.dataset, self.embedding_params, self.prediction_params
        ).to(self.device)

        self.logger.load_weights(cam_net.embedding_model, self.cfg.embedding_weights)
        self.logger.load_weights(cam_net.prediction_model, self.cfg.prediction_weights)

        criterion = nn.CrossEntropyLoss()
        return cam_net, criterion

    def generate_heatmaps(self, loader):
        model, criterion = self.get_model()

        features, gradients = [], []
        mri_shape = self.dataset.get_data_shape()[0][1:]
        num_views = 4
        num_pts = len(loader)

        def save_conv(module, input, output):
            features.append(output.detach())
            # features.copy_(output.detach())

        def save_grad(module, grad_input, grad_output):
            # gradients.copy_(F.relu(grad_output[0].detach()))
            gradients.insert(0, F.relu(grad_output[0].detach()))

        for i in range(num_views):
            model.embedding_model.model[2 * i].layers[0].register_full_backward_hook(
                save_grad
            )
            model.embedding_model.model[2 * i].layers[0].register_forward_hook(
                save_conv
            )

        heatmaps = torch.zeros(num_pts, num_views, 3, *mri_shape, requires_grad=False)
        orig_volumes = torch.zeros(num_pts, 3, *mri_shape, requires_grad=False)
        ptids = []

        for idx, (mat, clin_vars, ground_truth, ptid) in enumerate(loader):
            with torch.no_grad():
                volumes = torch.stack(
                    [
                        torch.stack(
                            [
                                util.load_scan(pt_tp, self.device)
                                for pt_tp in util.split_paths(pt)
                            ]
                        )
                        for pt in mat
                    ]
                )

                orig_volumes[idx, :] = volumes.detach().cpu().clone()

            bs, num_tp, d1, d2, d3 = volumes.shape

            model.zero_grad()
            volumes.requires_grad_()
            clin_vars.requires_grad_()

            raw_output = model(volumes, clin_vars)
            preds = torch.argmax(F.softmax(raw_output, dim=1), dim=1)

            one_hot_output = torch.FloatTensor(bs, 2).zero_().cuda()
            one_hot_output[0][ground_truth[0].item()] = 1

            raw_output.backward(gradient=one_hot_output)

            guided_grads = model.image_reconstruction.pop().data[:, 0]

            for view in range(num_views):
                f, g = features[view], gradients[view]
                weights = torch.mean(g, axis=(2, 3, 4))

                # TODO: might be able to shorten this a lot
                cam = torch.zeros(
                    (3, *f.shape[2:]), dtype=torch.float32, device=self.device
                )

                for tp_idx in range(3):
                    for i, w in enumerate(weights[tp_idx]):
                        cam[tp_idx] += F.relu(w * f[tp_idx, i, :, :, :])

                heatmaps[idx, view, :] = self.postprocess_heatmap(
                    cam, guided_grads, mri_shape
                ).cpu()

            features, gradients = [], []
            del raw_output

            ptids.extend(ptid)
            logging.info(f"Generated raw heatmap {idx+1}/{num_pts}")

        return heatmaps, orig_volumes, ptids

    def postprocess_heatmap(
        self,
        heatmap: torch.Tensor,
        guided_grads: torch.Tensor,
        target_shape: Tuple[int],
    ):
        tp, d1, d2, d3 = heatmap.shape

        heatmap = torch.stack(
            [
                F.interpolate(vol.view(1, 1, d1, d2, d3), size=target_shape).squeeze()
                for vol in heatmap
            ],
            dim=0,
        )

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        guided_grads = (guided_grads - guided_grads.min()) / (
            guided_grads.max() - guided_grads.min()
        )

        guided_heatmap = heatmap * guided_grads
        guided_heatmap = (guided_heatmap - guided_heatmap.min()) / (
            guided_heatmap.max() - guided_heatmap.min()
        )

        guided_heatmap[guided_heatmap < self.cfg.heatmap_min] = 0
        guided_heatmap[guided_heatmap > self.cfg.heatmap_max] = 1

        return guided_heatmap

    def get_heatmaps(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        if len(self.cfg.volume_path) == 0:
            if not isinstance(self.dataset.get_split_type(), dc.BasicSplit):
                logging.error(f"Heatmaps task have basic split")
                exit(1)

            if self.dataset.get_batch_size() != 1:
                logging.error("Heatmaps task expects a batch size of 1")
                exit(1)

            train_loader, val_loader, test_loader = self.dataset.get_data()[0]
            if val_loader is not None:
                logging.error("Heatmaps task expects val_loader to be None")
                exit(1)

            heatmaps, volumes, ptids = self.generate_heatmaps(test_loader)
            saved = self.logger.save_heatvol(heatmaps, volumes, ptids)
            print(f"Saved heatvols to {saved}")
        else:
            heatmaps, volumes, ptids = self.logger.load_heatvol(self.cfg.volume_path)

        return heatmaps, volumes, ptids

    def select_best_slices(
        self,
        cam: torch.Tensor,
        num_slices: int,
        view_num: int,
        thresh=0.5,
        simple=True,
    ) -> List[int]:
        simple_slices = [
            int((0.3 + (i / (num_slices - 1) * 0.4)) * cam.shape[view_num])
            for i in range(num_slices)
        ]

        target = thresh * cam.max()

        indices = torch.nonzero(cam > target, as_tuple=False)[:, view_num].unique()

        importance = torch.argsort(
            cam.index_select(view_num, indices)
            .sub(target)
            .relu()
            .sum(dim=tuple(i for i in range(cam.ndim) if i != view_num)),
            descending=True,
        )

        if len(importance) < num_slices:
            tmp = torch.zeros(num_slices)
            tmp[: len(importance)] = indices[importance]

        optim_slices = torch.msort(indices[importance[:num_slices]])
        for idx, val in enumerate(simple_slices):
            close = optim_slices[(optim_slices >= val - 5) & (optim_slices <= val + 5)]
            if len(close) != 0:
                simple_slices[idx] = close[0].item()

        if simple:
            return simple_slices
        else:
            return optim_slices

    def plot_heatmap_single(
        self,
        axes: List[plt.Axes],
        cam: torch.Tensor,
        orig: torch.Tensor,
        view_num: int,
        num_slices: int = 10,
        scan_alpha=0.7,
    ):
        rotations = [3, 3, 1]  # sagittal, coronal, axial
        slices = self.select_best_slices(cam, num_slices, view_num)
        for slice_idx, slice_num in enumerate(slices):
            ax = axes[slice_idx]

            cam_rot = np.rot90(cam.select(view_num, slice_num), rotations[view_num])
            orig_rot = np.rot90(orig.select(view_num, slice_num), rotations[view_num])

            cam_rot = gaussian_filter(cam_rot, self.cfg.gaussian_sigma)

            ax.axis("off")
            ax.tick_params(
                axis="both",
                left="off",
                top="off",
                right="off",
                bottom="off",
                labelleft="off",
                labeltop="off",
                labelright="off",
                labelbottom="off",
            )

            ax.imshow(orig_rot, cmap="gray", origin="lower")
            ax.imshow(
                cam_rot,
                cmap=COLORMAP,
                origin="lower",
                clim=[self.cfg.heatmap_min, self.cfg.heatmap_max],
                alpha=scan_alpha,
                interpolation="none",
            )

    def plot_heatmap_timepoints(
        self,
        heatmap: torch.Tensor,
        volume: torch.Tensor,
        ptid: str,
        num_slices: int = 10,
        conv_layer=0,
        planes=("Sagittal", "Coronal", "Axial"),
    ):
        num_conv_views, num_tp, *mri_dim = heatmap.shape

        if conv_layer >= num_conv_views:
            logging.error(
                f"conv_layer({conv_layer}) must be less than num_conv_views({num_conv_views})"
            )
            exit(1)

        if len(mri_dim) != len(planes):
            logging.error(
                f"mri dimension length({len(mri_dim)}) must be equal to the length of planes({len(planes)})"
            )
            exit(1)

        for plane_num, plane_name in enumerate(planes):
            fig = plt.figure(figsize=(10, 5), constrained_layout=True)
            gs = fig.add_gridspec(num_tp, num_slices, hspace=0, wspace=0)
            axes = gs.subplots(sharex="col", sharey="row")
            fig.suptitle(plane_name)

            for tp_idx in range(num_tp):
                self.plot_heatmap_single(
                    axes[tp_idx],
                    heatmap[conv_layer, tp_idx],
                    volume[tp_idx],
                    plane_num,
                    num_slices=num_slices,
                )

            self.logger.save_heatimage(f"{ptid}_{plane_name}")
            plt.close()

    def plot_heatmap_average(
        self,
        heatmap: torch.Tensor,
        volume: torch.Tensor,
        num_slices: int = 10,
        conv_layer=0,
        planes=("Sagittal", "Coronal", "Axial"),
    ):
        bs, num_conv_views, num_tp, *mri_dim = heatmap.shape

        if conv_layer >= num_conv_views:
            logging.error(
                f"conv_layer({conv_layer}) must be less than num_conv_views({num_conv_views})"
            )
            exit(1)

        if len(mri_dim) != len(planes):
            logging.error(
                f"mri dimension length({len(mri_dim)}) must be equal to the length of planes({len(planes)})"
            )
            exit(1)

        heatmap = torch.mean(heatmap, dim=(2))[:, conv_layer]
        heatmap = torch.where(heatmap > 0, 1, 0).sum(dim=0) / len(heatmap)
        volume = torch.mean(volume, dim=(0, 1))

        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(len(planes), num_slices, hspace=0, wspace=0)
        axes = gs.subplots(sharex="col", sharey="row")
        fig.suptitle("Average")

        for idx in range(len(planes)):
            self.plot_heatmap_single(
                axes[idx], heatmap, volume, idx, num_slices=num_slices
            )

        self.logger.save_heatimage(f"average")
        plt.close()

    def run(self):
        logging.info("Starting heatmaps task...")

        heatmaps, volumes, ptids = self.get_heatmaps()

        self.plot_heatmap_average(heatmaps, volumes)
        # for idx, (heatmap, volume, ptid) in enumerate(zip(heatmaps, volumes, ptids)):
        #     self.plot_heatmap_timepoints(heatmap, volume, ptid)
        #     print(f"[{idx+1}/{len(ptids)}] Saved heatmap image for {ptid}")

        logging.info("Done!")
