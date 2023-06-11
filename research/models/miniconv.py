import torch.nn as nn
import torch


# class MainModel(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()

#         num_layers = 1
#         hidden_size = out_dim
#         dropout = 0.0

#         self.rnn = nn.RNN(
#             input_size=in_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout,
#         )

#     def forward(self, x):
#         return self.rnn(x)[0][:, -1, :]


class MainModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        num_layers = 1
        hidden_size = 12
        dropout = 0.0

        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        return self.linear(self.rnn(x)[0][:, -1, :])


class MultiModalNet(nn.Module):
    def __init__(self, mri_shape, ni_shape, out_shape, **kwargs):
        super().__init__()

        self.model = MainModel(12 + ni_shape[-1], out_shape[0])

    def forward(self, x, cv):
        x = torch.nn.functional.adaptive_avg_pool3d(
            x.reshape(-1, 3, 12, 5, 6, 5), (1, 1, 1)
        ).reshape(-1, 3, 12)
        x = torch.cat([x, cv], dim=-1)

        return self.model(x)


class ImageOnly(nn.Module):
    def __init__(self, mri_shape, ni_shape, out_shape, **kwargs):
        super().__init__()

        self.model = MainModel(12, out_shape[0])

    def forward(self, x, cv):
        x = torch.nn.functional.adaptive_avg_pool3d(
            x.reshape(-1, 3, 12, 5, 6, 5), (1, 1, 1)
        ).reshape(-1, 3, 12)

        return self.model(x)


class CVOnly(nn.Module):
    def __init__(self, mri_shape, ni_shape, out_shape, **kwargs):
        super().__init__()

        self.model = MainModel(ni_shape[-1], out_shape[0])

    def forward(self, x, cv):
        return self.model(cv)
