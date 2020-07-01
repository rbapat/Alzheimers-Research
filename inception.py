'''
import torch.nn as nn
import numpy as np
import torch

class Conv3d(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(Conv3d, self).__init__()
        self.model = nn.Sequential(
                                        nn.Conv3d(in_features, out_features, **kwargs),
                                        nn.BatchNorm3d(out_features),
                                        nn.ReLU()
                                    )

    def forward(self, x):
        return self.model(x)

class InceptionNode(nn.Module):
    def __init__(self, in_features):
        super(InceptionNode, self).__init__()

        self.branch1 = nn.Sequential(
                                        nn.AvgPool3d(kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(in_features, 128, kernel_size = 1, stride = 1, padding = 0)
                                    )

        self.branch2 = nn.Sequential(
                                        Conv3d(in_features, 192, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(192, 384, kernel_size = 3, stride = 1, padding = 1)
                                    )

        self.branch3 = nn.Sequential(
                                        Conv3d(in_features, 192, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(192, 224, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(224, 224, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(224, 256, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0))
                                    )

        self.branch4 = nn.Sequential(
                                        Conv3d(in_features, 192, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(192, 192, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(192, 224, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(224, 224, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                        Conv3d(224, 224, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(224, 256, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(256, 256, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0))
                                    )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim = 1)
        
        return x

class InceptionStem(nn.Module):
    def __init__(self):
        super(InceptionStem, self).__init__()

        self.stem = nn.Sequential(
                                    Conv3d(1, 32, kernel_size = 3, stride = 2, padding = 1),
                                    Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(64, 64, kernel_size = 3, stride = 1, padding = 1)
                                )

        self.downsample_left = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        self.downsample_right = Conv3d(64, 96, kernel_size = 3, stride = 2, padding = 1)

        self.branch_left = nn.Sequential(
                                            Conv3d(160, 64, kernel_size = 1, stride = 1, padding = 0),
                                            Conv3d(64, 96, kernel_size = 3, stride = 1, padding = 1)
                                        )

        self.branch_right = nn.Sequential(
                                            Conv3d(160, 64, kernel_size = 1, stride = 1, padding = 0),
                                            Conv3d(64, 64, kernel_size = (1,1,7), stride = 1, padding = (0, 0, 3)),
                                            Conv3d(64, 64, kernel_size = (1,7,1), stride = 1, padding = (0, 3, 0)),
                                            Conv3d(64, 64, kernel_size = (7,1,1), stride = 1, padding = (3, 0, 0)),
                                            Conv3d(64, 96, kernel_size = 3, stride = 1, padding = 1)   
                                        )

        self.left = Conv3d(192, 192, kernel_size = 3, stride = 2, padding = 1)
        self.right = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        x = self.stem(x)

        left = self.downsample_left(x)
        right = self.downsample_right(x)

        x = torch.cat([left, right], dim = 1)

        left = self.branch_left(x)
        right = self.branch_right(x)

        x = torch.cat([left, right], dim = 1)

        left = self.left(x)
        right = self.right(x)

        x = torch.cat([left, right], dim = 1)

        return x

class InceptionReduce(nn.Module):
    def __init__(self, in_features):
        super(InceptionReduce, self).__init__()

        self.branch1 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)

        self.branch2 = nn.Sequential(
                                        Conv3d(in_features, 192, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(192, 192, kernel_size = 3, stride = 2, padding = 1)
                                    )

        self.branch3 = nn.Sequential(
                                        Conv3d(in_features, 256, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(256, 256, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(256, 320, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(320, 320, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                        Conv3d(320, 320, kernel_size = 3, stride = 2, padding = 1)
                                    )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b1, b2, b3], dim = 1)

        return x

class InceptionModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(InceptionModel, self).__init__()
        self.identifier = 'InceptionModel'
        self.dims = (in_depth, in_height, in_width)

        self.stem =InceptionStem()

        self.incept1 = InceptionNode(384)

        self.reduce1 = InceptionReduce(1024)

        self.incept2 = InceptionNode(1536)

        self.reduce2 = InceptionReduce(1024)

        self.drop = nn.Dropout3d(0.4)
        self.linear = nn.Linear(1536 * np.product(self.dims) // (2**5)**3, out_features)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x = self.stem(x)
        x = self.incept1(x)
        x = self.reduce1(x)
        x = self.incept2(x)
        x = self.reduce2(x)

        x = self.drop(x)

        x = torch.flatten(x, 1)

        x = self.linear(x)
        
        return x
'''


import torch.nn as nn
import numpy as np
import torch

class Conv3d(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(Conv3d, self).__init__()
        self.model = nn.Sequential(
                                        nn.Conv3d(in_features, out_features, **kwargs),
                                        nn.BatchNorm3d(out_features),
                                        nn.ReLU()
                                    )

    def forward(self, x):
        return self.model(x)

class InceptionNode(nn.Module):
    def __init__(self, in_features):
        super(InceptionNode, self).__init__()

        self.branch1 = nn.Sequential(
                                        nn.AvgPool3d(kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(in_features, 128, kernel_size = 1, stride = 1, padding = 0)
                                    )

        self.branch2 = nn.Sequential(
                                        Conv3d(in_features, 128, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(128, 256, kernel_size = 3, stride = 1, padding = 1)
                                    )

        self.branch3 = nn.Sequential(
                                        Conv3d(in_features, 32, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(32, 64, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(64, 64, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(64, 64, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0))
                                    )

        self.branch4 = nn.Sequential(
                                        Conv3d(in_features, 32, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(32, 32, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(32, 64, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(64, 64, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                        Conv3d(64, 64, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(64, 64, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(64, 64, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0))
                                    )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim = 1)
        
        return x

class InceptionStem(nn.Module):
    def __init__(self):
        super(InceptionStem, self).__init__()

        self.stem = nn.Sequential(
                                    Conv3d(1, 8, kernel_size = 3, stride = 2, padding = 1),
                                    Conv3d(8, 16, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(16, 16, kernel_size = 3, stride = 1, padding = 1)
                                )

        self.downsample_left = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        self.downsample_right = Conv3d(16, 32, kernel_size = 3, stride = 2, padding = 1)

        self.branch_left = nn.Sequential(
                                            Conv3d(48, 32, kernel_size = 1, stride = 1, padding = 0),
                                            Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1)
                                        )

        self.branch_right = nn.Sequential(
                                            Conv3d(48, 32, kernel_size = 1, stride = 1, padding = 0),
                                            Conv3d(32, 32, kernel_size = (1,1,7), stride = 1, padding = (0, 0, 3)),
                                            Conv3d(32, 32, kernel_size = (1,7,1), stride = 1, padding = (0, 3, 0)),
                                            Conv3d(32, 32, kernel_size = (7,1,1), stride = 1, padding = (3, 0, 0)),
                                            Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1)   
                                        )

        self.left = Conv3d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.right = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        x = self.stem(x)

        left = self.downsample_left(x)
        right = self.downsample_right(x)

        x = torch.cat([left, right], dim = 1)

        left = self.branch_left(x)
        right = self.branch_right(x)

        x = torch.cat([left, right], dim = 1)

        left = self.left(x)
        right = self.right(x)

        x = torch.cat([left, right], dim = 1)

        return x

class InceptionReduce(nn.Module):
    def __init__(self, in_features):
        super(InceptionReduce, self).__init__()

        self.branch1 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)

        self.branch2 = nn.Sequential(
                                        Conv3d(in_features, 128, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(128, 128, kernel_size = 3, stride = 2, padding = 1)
                                    )

        self.branch3 = nn.Sequential(
                                        Conv3d(in_features, 128, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(128, 128, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(128, 256, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(256, 256, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                        Conv3d(256, 256, kernel_size = 3, stride = 2, padding = 1)
                                    )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b1, b2, b3], dim = 1)

        return x

class InceptionModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(InceptionModel, self).__init__()
        self.identifier = 'InceptionModel'
        self.dims = (in_depth, in_height, in_width)

        self.stem =InceptionStem()

        self.incept1 = InceptionNode(256)
        self.incept2 = InceptionNode(512)
        self.incept3 = InceptionNode(512)

        self.reduce1 = InceptionReduce(512)

        self.incept4 = InceptionNode(896)
        self.incept5 = InceptionNode(512)
        self.incept6 = InceptionNode(512)
        self.incept7 = InceptionNode(512)
        self.incept8 = InceptionNode(512)
        self.incept9 = InceptionNode(512)
        self.incept10 = InceptionNode(512)

        self.reduce2 = InceptionReduce(512)

        self.drop = nn.Dropout3d(0.4)
        self.linear = nn.Linear(896 * np.product(self.dims) // (2**5)**3, out_features)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x = self.stem(x)
        x = self.incept1(x)
        x = self.incept2(x)
        x = self.incept3(x)
        x = self.reduce1(x)
        x = self.incept4(x)
        x = self.incept5(x)
        x = self.incept6(x)
        x = self.incept7(x)
        x = self.incept8(x)
        x = self.incept9(x)
        x = self.incept10(x)
        x = self.reduce2(x)

        x = self.drop(x)

        x = torch.flatten(x, 1)

        x = self.linear(x)
        
        return x    