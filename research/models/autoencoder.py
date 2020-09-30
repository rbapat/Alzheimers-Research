import torch.nn as nn
import skimage.transform
import numpy as np
import torch

class Conv3d(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(Conv3d, self).__init__()
        self.model = nn.Sequential(
                                        nn.Conv3d(in_features, out_features, bias=False, **kwargs),
                                        nn.BatchNorm3d(out_features),
                                        nn.ReLU()
                                    )

    def forward(self, x):
        return self.model(x)

class ConvTranspose3d(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(ConvTranspose3d, self).__init__()
        self.model = nn.Sequential(
                                        nn.ConvTranspose3d(in_features, out_features, bias=False, **kwargs),
                                        nn.BatchNorm3d(out_features),
                                        nn.ReLU()
                                    )

    def forward(self, x):
        return self.model(x)    



class AEStem(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(AEStem, self).__init__()
        self.identifier = 'AEStem'
        self.dims = (in_depth, in_height, in_width)
        
        self.conv1 = nn.Sequential(
                                    Conv3d(1, 16, kernel_size = 3, stride = 2, padding = 1),
                                    Conv3d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(16, 32, kernel_size = 3, stride = 1, padding = 1),
                                    )
        self.pool1 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1, return_indices = True)

        self.conv2 = nn.Sequential(
                                    Conv3d(32, 48, kernel_size = 1, stride = 1, padding = 0),
                                    Conv3d(48, 64, kernel_size = 3, stride = 1, padding = 1),
                                )

        self.pool2 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1, return_indices = True)

        
        self.t_pool2 = nn.MaxUnpool3d(kernel_size = 3, stride = 2, padding = 1)

        self.t_conv2 = nn.Sequential(
                                    ConvTranspose3d(64, 48, kernel_size = 3, stride = 1, padding = 1),
                                    ConvTranspose3d(48, 32, kernel_size = 1, stride = 1, padding = 0)
                                )

        self.t_pool1 = nn.MaxUnpool3d(kernel_size = 3, stride = 2, padding = 1)

        self.t_conv1 = nn.Sequential(
                                    ConvTranspose3d(32, 16, kernel_size = 3, stride = 1, padding = 1),
                                    ConvTranspose3d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                                    ConvTranspose3d(16, 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
                                )

    def forward(self, x):
        masks = []

        x = x.view(-1, 1, *self.dims)

        x = self.conv1(x)
        sz1 = x.shape
        x, idx1 = self.pool1(x)
        x = self.conv2(x)
        sz2 = x.shape
        x, idx2 = self.pool2(x)

        x = self.t_pool2(x, idx2, output_size = sz2)
        x = self.t_conv2(x)
        x = self.t_pool1(x, idx1, output_size = sz1)
        x = self.t_conv1(x)

        return x