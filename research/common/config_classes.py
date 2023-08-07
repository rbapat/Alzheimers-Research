from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch


@dataclass
class TrainConfig:
    model_cls: type

    optim: partial
    loss_function: torch.nn.Module

    num_epochs: int
    es_tolerance: int

    model_weights: Optional[str] = None


@dataclass
class EmbeddingConfig:
    scan_path: str
    embedding_path: str
    weight_path: str
    model_cls: type


@dataclass
class HeatmapsConfig:
    embedding_model_cls: type
    embedding_weights: str

    prediction_model_cls: type
    prediction_weights: str

    heatmap_min: float
    heatmap_max: float
    gaussian_sigma: int

    volume_path: Optional[str] = ""
