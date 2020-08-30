import torch.nn as nn
import skimage.transform
import numpy as np
import torch

class EModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(EModel, self).__init__()
        self.identifier = 'EModel'
        self.dims = (in_depth, in_height, in_width)

        self.conv1 = nn.Sequential(
                                    Conv3d(1, 16, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0)
                                )

        self.conv2 = nn.Sequential(
                                    Conv3d(16, 32, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(32, 32, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0)
                                )

        self.conv3 = nn.Sequential(
                                    Conv3d(32, 32, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0)
                                )

        self.conv4 = nn.Sequential(
                                    Conv3d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(64, 128, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(128, 128, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0)
                                )

        self.conv5 = nn.Sequential(
                                    Conv3d(128, 128, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(128, 128, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(128, 128, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0)
                                )

        self.conv_linear_features = (128 * in_depth * in_height * in_width) // (2**5)**3
        self.dense = nn.Sequential(
                                    nn.Linear(self.conv_linear_features, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(0.6),
                                    nn.Linear(4096, 2048),
                                    nn.Dropout(0.3),
                                    nn.ReLU(),
                                    nn.Linear(2048, out_features)
                                )


        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(-1, self.conv_linear_features)

        x = self.dense(x)

        return x