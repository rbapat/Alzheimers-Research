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

    model_weights: Optional[str] = None
