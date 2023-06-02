import logging
import random
import sys
import os

from omegaconf import DictConfig
import numpy as np
import hydra
import torch

from research.common.config_classes import BaseConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    kwargs = dict(cfg["kwargs"])
    set_seed(kwargs["seed"])

    kwargs["logging"] = hydra.utils.instantiate(cfg.logging, **kwargs)
    kwargs["task"] = hydra.utils.instantiate(cfg.task, **kwargs)

    kwargs["task"].run()


if __name__ == "__main__":
    main()
