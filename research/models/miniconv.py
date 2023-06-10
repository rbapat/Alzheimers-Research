import torch.nn as nn
import torch


# bs: 32, lr: 1e-3
class MultiModalNet(nn.Module):
    def __init__(self, mri_shape, ni_shape, **kwargs):
        super().__init__()

        num_layers = 1
        hidden_size = 2
        dropout = 0.0

        self.rnn = nn.RNN(
            input_size=12 + ni_shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # self.linear = nn.Sequential(
        #     nn.Linear(hidden_size, 2),
        # )

    def forward(self, x, cv):
        x = torch.nn.functional.adaptive_avg_pool3d(
            x.reshape(-1, 3, 12, 5, 6, 5), (1, 1, 1)
        ).reshape(-1, 3, 12)
        x = torch.cat([x, cv], dim=-1)

        # x, _ = self.rnn(x)
        # x = self.linear(x[:, -1, :])
        return self.rnn(x)[0][:, -1, :]


class ImageOnly(nn.Module):
    def __init__(self, mri_shape, ni_shape, **kwargs):
        super().__init__()

        num_layers = 1
        hidden_size = 2
        dropout = 0.0

        self.rnn = nn.RNN(
            input_size=12,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # self.linear = nn.Sequential(
        #     nn.Linear(hidden_size, 2),
        # )

    def forward(self, x, cv):
        x = torch.nn.functional.adaptive_avg_pool3d(
            x.reshape(-1, 3, 12, 5, 6, 5), (1, 1, 1)
        ).reshape(-1, 3, 12)

        # x, _ = self.rnn(x)
        # x = self.linear(x[:, -1, :])
        return self.rnn(x)[0][:, -1, :]


class CVOnly(nn.Module):
    def __init__(self, mri_shape, ni_shape, **kwargs):
        super().__init__()

        num_layers = 1
        hidden_size = 2
        dropout = 0.0

        self.rnn = nn.RNN(
            input_size=ni_shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # self.linear = nn.Sequential(
        #     nn.Linear(hidden_size, 2),
        # )

    def forward(self, x, cv):
        # x, _ = self.rnn(x)
        # x = self.linear(x[:, -1, :])
        return self.rnn(cv)[0][:, -1, :]


# class ImageOnly(nn.Module):
#     def __init__(self, mri_shape, **kwargs):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=mri_shape[-1],
#                 out_channels=512,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=True,
#             ),
#             nn.ReLU(),
#             nn.Conv1d(
#                 in_channels=512,
#                 out_channels=256,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=True,
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
#             nn.Flatten(1),
#             nn.Linear(256, 2),
#         )

#         num_layers = 1
#         hidden_size = 256
#         dropout = 0.0

#         self.rnn = nn.RNN(
#             input_size=1800,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout,
#         )

#         self.linear = nn.Sequential(nn.Linear(hidden_size, 2))

#     def forward(self, x, cv):
#         # x = x.transpose(1, 2)
#         # return self.model(x)

#         x, _ = self.rnn(x)
#         x = self.linear(x[:, -1, :])
#         return x


# class CVOnly(nn.Module):
#     def __init__(self, ni_shape, **kwargs):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=ni_shape[-1],
#                 out_channels=512,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=True,
#             ),
#             nn.ReLU(),
#             nn.Conv1d(
#                 in_channels=512,
#                 out_channels=256,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=True,
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
#             nn.Flatten(1),
#             nn.Linear(256, 2),
#         )

#     def forward(self, x, cv):
#         cv = cv.transpose(1, 2)
#         return self.model(cv)
