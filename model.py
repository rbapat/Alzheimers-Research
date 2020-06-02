import torch.nn as nn
import numpy as np
import torch

class ADModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(ADModel, self).__init__()
        self.identifier = 'Baseline'
        self.dims = (in_depth, in_height, in_width)

        self.begin_norm = nn.BatchNorm3d(1)
        self.block1 = self.elementary_block(1, 8)
        self.block2 = self.elementary_block(8, 8)
        self.block3 = self.elementary_block(8, 16)
        self.block4 = self.elementary_block(16, 16)
        self.block5 = self.elementary_block(16, 32)
        self.conv1 = nn.Conv3d(32, 32, kernel_size = 3, stride = 1, padding = 1)
        self.end_norm = nn.BatchNorm3d(32)
        self.rect1 = nn.LeakyReLU(0.001)

        self.linear = nn.Linear(32 * in_width * in_height * in_depth // (2**5)**3, 2048)
        self.rect2 = nn.LeakyReLU(0.001)
        self.classifier = nn.Linear(2048, out_features)

    def elementary_block(self, in_features, out_features):
        return nn.Sequential(
                                nn.Conv3d(in_features, out_features, kernel_size = 3, stride = 1, padding = 1),
                                nn.BatchNorm3d(out_features),
                                nn.LeakyReLU(0.001),
                                nn.AvgPool3d(kernel_size = 3, stride = 2, padding = 1),
                            )

    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x = self.begin_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.conv1(x)
        x = self.end_norm(x)
        x = self.rect1(x)

        x = x.view(-1, 32 * self.dims[0] * self.dims[1] * self.dims[2] // (2**5)**3)
        x = self.linear(x)

        x = self.rect2(x)

        x = self.classifier(x)

        return x


class ExperimentalModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(ExperimentalModel, self).__init__()
        self.identifier = 'Experimental'
        self.dims = (in_depth, in_height, in_width)

        self.begin_norm = nn.BatchNorm3d(1)
        self.intro1 = self.intro_block(1, 16)
        self.intro2 = self.intro_block(16, 32)

        self.middle1 = self.middle_block(32, 32)
        self.middle2 = self.middle_block(32, 64)

        self.end = self.middle_block(64, 64)
        self.conv = nn.Conv3d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.end_norm = nn.BatchNorm3d(64)
        self.rect = nn.LeakyReLU(0.001)

        self.linear = nn.Linear(64 * in_width * in_height * in_depth // (2**5)**3, 2048)
        self.rect1 = nn.LeakyReLU(0.001)
        self.classifier = nn.Linear(2048, out_features)

    def intro_block(self, in_features, out_features):
        return nn.Sequential(
                                nn.Conv3d(in_features, out_features, kernel_size = 5, stride = 1, padding = 2),
                                nn.BatchNorm3d(out_features),
                                nn.LeakyReLU(0.001),
                                nn.AvgPool3d(kernel_size = 2, stride = 2, padding = 0)
            )

    def middle_block(self, in_features, out_features):
        return nn.Sequential(
                                nn.Conv3d(in_features, out_features, kernel_size = 3, stride = 1, padding = 1),
                                nn.BatchNorm3d(out_features),
                                nn.LeakyReLU(0.001),
                                nn.AvgPool3d(kernel_size = 3, stride = 2, padding = 1),
                            )


    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x = self.begin_norm(x)

        x = self.intro1(x)
        x = self.intro2(x)

        x = self.middle1(x)
        x = self.middle2(x)

        x = self.end(x)
        x = self.conv(x)
        x = self.end_norm(x)
        x = self.rect(x)

        x = x.view(-1, 64 * self.dims[0] * self.dims[1] * self.dims[2] // (2**5)**3)
        x = self.linear(x)
        x = self.rect1(x)
        x = self.classifier(x)

        return x


        return x
