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

#256
class InceptionNodeB(nn.Module):
    def __init__(self, in_features):
        super(InceptionNodeB, self).__init__()

        self.branch1 = nn.Sequential(
                                        nn.AvgPool3d(kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(in_features, 64, kernel_size = 1, stride = 1, padding = 0)
                                    )

        self.branch2 = Conv3d(in_features, 64, kernel_size = 1, stride = 1, padding = 0)

        self.branch3 = nn.Sequential(
                                        Conv3d(in_features, 32, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1)
                                    )

        self.branch4 = nn.Sequential(
                                        Conv3d(in_features, 32, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(64, 64, kernel_size = 3, stride = 1, padding = 1)
                                    )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim = 1)
    
        return x 

#224 + in
class InceptionReduceB(nn.Module):
    def __init__(self, in_features):
        super(InceptionReduceB, self).__init__()

        self.branch1 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)

        self.branch2 = Conv3d(in_features, 128, kernel_size = 3, stride = 2, padding = 1)

        self.branch3 = nn.Sequential(
                                        Conv3d(in_features, 32, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(64, 96, kernel_size = 3, stride = 2, padding = 1)
                                    )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b1, b2, b3], dim = 1)

        return x     

#512
class InceptionNodeC(nn.Module):
    def __init__(self, in_features):
        super(InceptionNodeC, self).__init__()

        self.branch1 = nn.Sequential(
                                        nn.AvgPool3d(kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(in_features, 64, kernel_size = 3, stride = 1, padding = 1)
                                    )

        self.branch2 = Conv3d(in_features, 64, kernel_size = 3, stride = 1, padding = 1)

        self.branch3_stem = Conv3d(in_features, 96, kernel_size = 3, stride = 1, padding = 1)
        self.branch3_left = Conv3d(96, 64, kernel_size = (1, 1, 3), stride = 1, padding = (0, 0, 1))
        self.branch3_middle = Conv3d(96, 64, kernel_size = (1, 3, 1), stride = 1, padding = (0, 1, 0))
        self.branch3_right = Conv3d(96, 64, kernel_size = (3, 1, 1), stride = 1, padding = (1, 0, 0))

        self.branch4_stem = nn.Sequential(
                                            Conv3d(in_features, 96, kernel_size = 3, stride = 1, padding = 1),
                                            Conv3d(96, 96, kernel_size = (1, 1, 3), stride = 1, padding = (0, 0, 1)),
                                            Conv3d(96, 128, kernel_size = (1, 3, 1), stride = 1, padding = (0, 1, 0)),
                                            Conv3d(128, 128, kernel_size = (3, 1, 1), stride = 1, padding = (1, 0, 0))
                                        )

        self.branch4_left = Conv3d(128, 64, kernel_size = (1, 1, 3), stride = 1, padding = (0, 0, 1))
        self.branch4_middle = Conv3d(128, 64, kernel_size = (1, 3, 1), stride = 1, padding = (0, 1, 0))
        self.branch4_right = Conv3d(128, 64, kernel_size = (3, 1, 1), stride = 1, padding = (1, 0, 0))

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        b3 = self.branch3_stem(x)
        b3_l = self.branch3_left(b3)
        b3_m = self.branch3_middle(b3)
        b3_r = self.branch3_right(b3)

        b4 = self.branch4_stem(x)
        b4_l = self.branch4_left(b4)
        b4_m = self.branch4_middle(b4)
        b4_r = self.branch4_right(b4)

        x = torch.cat([b1, b2, b3_l, b3_m, b3_r, b4_l, b4_m, b4_r], dim = 1)

        return x

#192 + in
class InceptionReduceC(nn.Module):
    def __init__(self, in_features):
        super(InceptionReduceC, self).__init__()

        self.branch1 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)

        self.branch2 = nn.Sequential(
                                        Conv3d(in_features, 64, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(64, 64, kernel_size = 3, stride = 2, padding = 1)
                                    )

        self.branch3 = nn.Sequential(
                                        Conv3d(in_features, 96, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(96, 96, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                        Conv3d(96, 128, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                        Conv3d(128, 128, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                        Conv3d(128, 128, kernel_size = 3, stride = 2, padding = 1)
                                    )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b1, b2, b3], dim = 1)

        return x

#256
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

#384 + in
class InceptionReduceA(nn.Module):
    def __init__(self, in_features):
        super(InceptionReduceA, self).__init__()

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

#448
class InceptionNodeA(nn.Module):
    def __init__(self, in_features):
        super(InceptionNodeA, self).__init__()

        self.branch1 = nn.Sequential(
                                        nn.AvgPool3d(kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(in_features, 64, kernel_size = 1, stride = 1, padding = 0)
                                    )

        self.branch2 = Conv3d(in_features, 256, kernel_size = 1, stride = 1, padding = 0)

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

class InceptionModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, nonimaging_features, out_features):
        super(InceptionModel, self).__init__()
        self.identifier = 'Pretrain'
        self.dims = (in_depth, in_height, in_width)

        self.stem = InceptionStem()

        self.incept1 = InceptionNodeA(256)

        self.reduce1 = InceptionReduceA(448)
        
        self.incept2 = InceptionNodeB(832)
        self.incept3 = InceptionNodeB(256)
        self.incept4 = InceptionNodeB(256)
        self.incept5 = InceptionNodeB(256)
        self.incept6 = InceptionNodeB(256)

        self.reduce2 = InceptionReduceA(256)
        self.end_pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.drop = nn.Dropout3d(0.4)
        self._imaging_linear = nn.Linear(640, 1024)

        self._nonimaging_linear = nn.Linear(nonimaging_features, 1024)

        self._linear = nn.Linear(2048, out_features)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, non_imaging):
        x = x.view(-1, 1, *self.dims)

        x = self.stem(x)

        x = self.incept1(x)

        x = self.reduce1(x)
        
        x = self.incept2(x)
        x = self.incept3(x)
        x = self.incept4(x)
        x = self.incept5(x)
        x = self.incept6(x)

        x = self.reduce2(x)
        x = self.end_pool(x)

        x = self.drop(x)

        x = torch.flatten(x, 1)

        im = self._imaging_linear(x)
        non_im = self._nonimaging_linear(non_imaging)

        x = torch.cat([im, non_im], dim = 1)
        x = self._linear(x)
        
        return x    

