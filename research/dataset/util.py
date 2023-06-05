from typing import List

import nibabel as nib
import torch

SEP = "###"


def load_scan(path: str, device: torch.device) -> torch.Tensor:
    mat = nib.load(path).get_fdata()
    mat = (mat - mat.min()) / (mat.max() - mat.min())  # min-max normalization
    return torch.tensor(mat, device=device, dtype=torch.float)


def join_paths(paths: List[str]) -> str:
    return SEP.join(paths)


def split_paths(paths: str) -> List[str]:
    return paths.split(SEP)
