import torch.nn as nn
import torch


class MultiModalNet(nn.Module):
    def __init__(self, mri_shape, ni_shape, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=mri_shape[-1] + ni_shape[-1],
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(1),
            nn.Linear(256, 2),
        )

    def forward(self, x, cv):
        x = torch.cat([x, cv], dim=-1)
        x = x.transpose(1, 2)
        return self.model(x)


class ImageOnly(nn.Module):
    def __init__(self, mri_shape, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=mri_shape[-1],
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(1),
            nn.Linear(256, 2),
        )

    def forward(self, x, cv):
        x = x.transpose(1, 2)
        return self.model(x)


class CVOnly(nn.Module):
    def __init__(self, ni_shape, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=ni_shape[-1],
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(1),
            nn.Linear(256, 2),
        )

    def forward(self, x, cv):
        cv = cv.transpose(1, 2)
        return self.model(cv)
