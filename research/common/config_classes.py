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
