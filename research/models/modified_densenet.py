import skimage.transform
import skimage.color
import torch.nn as nn
import numpy as np
import torch
import math

class Conv3d(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(Conv3d, self).__init__()
        self.model = nn.Sequential(
                                        nn.BatchNorm3d(in_features),
                                        nn.ReLU(),
                                        nn.Conv3d(in_features, out_features, bias=True, **kwargs)
                                    )

    def forward(self, x):
        return self.model(x)

class DenseUnit(nn.Module):
    def __init__(self, in_features, growth_rate):
        super(DenseUnit, self).__init__()

        self.bottleneck = Conv3d(in_features, 4 * growth_rate, kernel_size = 1, stride = 1, padding = 0)
        self.conv2 = Conv3d(4 * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.conv2(x)

        return x

class DenseBlock(nn.Module):
    def __init__(self, in_features, num_layers, growth_rate):
        super(DenseBlock, self).__init__()

        layers = [DenseUnit(in_features, growth_rate)]
        for i in range(1, num_layers):
            layers.append(DenseUnit(in_features + i * growth_rate, growth_rate))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for idx, bottleneck in enumerate(self.layers):
            new_x = bottleneck(x)

            if idx < len(self.layers) - 1:
                x = torch.cat([new_x, x], dim = 1)
            else:
                x = new_x

        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_features, theta):
        super(TransitionBlock, self).__init__()

        self.conv = Conv3d(in_features, int(in_features * theta), kernel_size = 1, stride = 1, padding = 0)
        self.pool = nn.AvgPool3d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)

        return x

class DenseNet(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(DenseNet, self).__init__()
        self.identifier = 'DenseNet'
        self.dims = (in_depth, in_height, in_width)     
        
        growth_rate = 12
        theta = 1.0

        compressed_size = int(growth_rate * theta)

        self.stem = nn.Sequential(
                                    Conv3d(1, 2 * growth_rate, kernel_size = 7, stride = 2, padding = 3),
                                    nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
                                )

        self.dense1 = DenseBlock(2 * growth_rate, 6, growth_rate)
        self.trans1 = TransitionBlock(growth_rate, theta)

        self.dense2 = DenseBlock(compressed_size, 12, growth_rate)
        self.trans2 = TransitionBlock(growth_rate, theta)

        self.dense3 = DenseBlock(compressed_size, 24, growth_rate)
        self.trans3 = TransitionBlock(growth_rate, theta)

        self.dense4 = DenseBlock(compressed_size, 16, growth_rate)

        self.end_pool = nn.AdaptiveAvgPool3d((1,1,1))
        
        self.drop = nn.Dropout(0.7)
        self.fc = nn.Linear(growth_rate, out_features)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x = self.stem(x)

        x = self.dense1(x)
        x = self.trans1(x)

        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)

        x = self.dense4(x)

        x = self.end_pool(x)

        x = torch.flatten(x, 1)

        x = self.drop(x)
        x = self.fc(x)

        return x