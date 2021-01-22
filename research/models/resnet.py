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
                                        nn.Conv3d(in_features, out_features, bias = False, **kwargs),
                                        nn.BatchNorm3d(in_features),
                                        nn.ReLU()
                                    )

    def forward(self, x):
        return self.model(x)

class Resnet(nn.Module):
    def __init__(self, in_dims, out_features, channels):
        super(Resnet, self).__init__()
        self.identifier = 'Resnet'
        self.dims = in_dims

        self.stem = nn.Sequential(
                                    Conv3d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                                    nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
                                )

        layers = [ResidualBlock(channels[0], False)]
        for idx, channel in enumerate(channels[1:]):
            layers.append(ResidualBlock(channels[idx], True))

        self.model = nn.Sequential(*layers)
      
    def forward(self, x):
        x = x.view(-1, 1, *self.dims)


        return x

    def init_optimizer(self):
        optim =  torch.optim.SGD(self.parameters(), lr = 0.001, momentum = 0.9, weight_decay = .001, nesterov = True)
        scheduler = None 

        return optim, scheduler