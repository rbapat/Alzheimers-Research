import torch.nn as nn
import torch.nn.functional as F
import torch
import os


# Wrapper for convolution operations used in DenseNet
class Conv3d(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(Conv3d, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_features, out_features, bias=False, **kwargs),
            nn.BatchNorm3d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


# DenseUnit that essentially the DenseNet convolution operation
class DenseUnit(nn.Module):
    def __init__(self, in_features, growth_rate, drop_rate):
        super(DenseUnit, self).__init__()

        self.bottleneck = Conv3d(
            in_features, 4 * growth_rate, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = Conv3d(
            4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1
        )
        self.drop = nn.Dropout3d(drop_rate)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.conv2(x)
        x = self.drop(x)

        return x


# Block of densely connected DenseUnit (convolutional) operations
class DenseBlock(nn.Module):
    def __init__(self, in_features, num_layers, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()

        layers = [DenseUnit(in_features, growth_rate, drop_rate)]
        for i in range(1, num_layers):
            layers.append(
                DenseUnit(in_features + i * growth_rate, growth_rate, drop_rate)
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for idx, bottleneck in enumerate(self.layers):
            new_x = bottleneck(x)

            if idx < len(self.layers) - 1:
                x = torch.cat([new_x, x], dim=1)
            else:
                x = new_x

        return x


# Transition block for dimensionality reduction between DenseBlocks
class TransitionBlock(nn.Module):
    def __init__(self, in_features, theta, drop_rate):
        super(TransitionBlock, self).__init__()

        self.conv = Conv3d(
            in_features, int(in_features * theta), kernel_size=1, stride=1, padding=0
        )
        self.norm = nn.BatchNorm3d(int(in_features * theta))
        self.drop = nn.Dropout3d(drop_rate)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.pool(x)

        return x


# Main DenseNet implementation I'm using right now, implementation was based on the official pytorch implementation
class DenseNet(nn.Module):
    def __init__(
        self, mri_shape, out_shape, channels, growth_rate, theta, drop_rate, **kwargs
    ):
        super(DenseNet, self).__init__()

        self.dims = mri_shape

        compressed_size = int(growth_rate * theta)

        self.stem = nn.Sequential(
            Conv3d(1, 2 * growth_rate, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=2, stride=4, padding=0),
        )

        layers = [DenseBlock(2 * growth_rate, channels[0], growth_rate, drop_rate)]  # 0
        for idx, channel in enumerate(channels[1:]):
            layers.append(TransitionBlock(growth_rate, theta, drop_rate))  # 1 3 5
            layers.append(
                DenseBlock(compressed_size, channel, growth_rate, drop_rate)
            )  # 2 4 6

        self.model = nn.Sequential(*layers)

        self.end_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(growth_rate, out_shape[0])

    def features(self, x):
        x = x.view(-1, 1, *self.dims)
        x = self.stem(x)
        x = self.model(x)
        x = x.view(len(x), -1)
        return x

    def forward(self, x, clin_vars):
        x = x.view(-1, 1, *self.dims)

        x = self.stem(x)

        x = self.model(x)

        x = self.end_pool(x).view(len(x), -1)
        x = self.fc(x)

        return x
