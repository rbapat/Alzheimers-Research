import torch.nn as nn
import torch


class MainModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=0),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=0),
            nn.Flatten(1),
            nn.Linear(64 * 3, out_dim),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.model(x)


class MultiModalNet(nn.Module):
    def __init__(self, mri_shape, ni_shape, out_shape, **kwargs):
        super().__init__()

        self.model = MainModel(mri_shape[-1] + ni_shape[-1], out_shape[0])

    def forward(self, x, cv):
        x = torch.cat([x, cv], dim=-1)
        return self.model(x)


class ImageOnly(nn.Module):
    def __init__(self, mri_shape, ni_shape, out_shape, **kwargs):
        super().__init__()

        self.model = MainModel(mri_shape[-1], out_shape[0])

    def forward(self, x, cv):
        return self.model(x)


class CVOnly(nn.Module):
    def __init__(self, mri_shape, ni_shape, out_shape, **kwargs):
        super().__init__()

        self.model = MainModel(ni_shape[-1], out_shape[0])

    def forward(self, x, cv):
        return self.model(cv)
