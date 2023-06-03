import random

import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


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
