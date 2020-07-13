import torch.nn as nn
import numpy as np
import torch

class EmptyNode(nn.Module):
    def __init__(self):
        super(EmptyNode, self).__init__()

    def forward(self, x):
        return x

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

class AEConv3d(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(AEConv3d, self).__init__()
        self.model = nn.Sequential(
                                        nn.ConvTranspose3d(in_features, out_features, **kwargs),
                                        nn.BatchNorm3d(out_features),
                                        nn.ReLU()
                                    )

    def forward(self, x):
        return self.model(x)

class InceptionNode(nn.Module):
    def __init__(self, in_features):
        super(InceptionNode, self).__init__()

        self.branch1_p = nn.MaxPool3d(kernel_size = 3, stride = 1, padding = 1, return_indices = True)
        self.branch1 = nn.Sequential(
                                        EmptyNode(),
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
        pooled1, idx = self.branch1_p(x)

        b1 = self.branch1(pooled1)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim = 1)
        
        return x, idx

class AEInceptionNode(nn.Module):
    def __init__(self, in_features):
        super(AEInceptionNode, self).__init__()

        self.inv_branch1 = nn.Sequential(
                                            AEConv3d(128, in_features, kernel_size = 1, stride = 1, padding = 0),
                                            EmptyNode()
                                        )
        self.inv_branch1_p = nn.MaxUnpool3d(kernel_size = 3, stride = 1, padding = 1)

        self.inv_branch2 = nn.Sequential(
                                            AEConv3d(256, 128, kernel_size = 3, stride = 1, padding = 1),
                                            AEConv3d(128, in_features, kernel_size = 1, stride = 1, padding = 0)
                                        )

        self.inv_branch3 = nn.Sequential(
                                            AEConv3d(64, 64, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                            AEConv3d(64, 64, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                            AEConv3d(64, 32, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                            AEConv3d(32, in_features, kernel_size = 1, stride = 1, padding = 0)
                                        )

        self.inv_branch4 = nn.Sequential(
                                            AEConv3d(64, 64, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                            AEConv3d(64, 64, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                            AEConv3d(64, 64, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                            AEConv3d(64, 64, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                            AEConv3d(64, 32, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                            AEConv3d(32, 32, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                            AEConv3d(32, in_features, kernel_size = 1, stride = 1, padding = 0),
                                        )

    def forward(self, x, idx):
        branch1, branch2, branch3, branch4 = torch.split(x, [128, 256, 64, 64], dim = 1)

        branch1 = self.inv_branch1(x)
        branch1 = self.inv_branch1_p(branch1, idx)

        branch2 = self.inv_branch2(x)
        branch3 = self.inv_branch3(x)
        branch4 = self.inv_branch4(x)

        x = torch.mean(torch.stack([branch1, branch2, branch3, branch4]))

        return x

class InceptionStem(nn.Module):
    def __init__(self):
        super(InceptionStem, self).__init__()

        self.stem = nn.Sequential(
                                    Conv3d(1, 8, kernel_size = 3, stride = 2, padding = 1),
                                    Conv3d(8, 16, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(16, 16, kernel_size = 3, stride = 1, padding = 1)
                                )

        self.downsample_left = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1, return_indices = True)
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
        self.right = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1, return_indices = True)

    def forward(self, x):
        x = self.stem(x)

        left, idx1 = self.downsample_left(x)
        right = self.downsample_right(x)

        x = torch.cat([left, right], dim = 1)

        left = self.branch_left(x)
        right = self.branch_right(x)

        x = torch.cat([left, right], dim = 1)

        left = self.left(x)
        right, idx2 = self.right(x)

        x = torch.cat([left, right], dim = 1)

        return x, idx1, idx2

class AEInceptionStem(nn.Module):
    def __init__(self, in_features):
        super(AEInceptionStem, self).__init__()

        self.inv_left = AEConv3d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.inv_right = nn.MaxUnpool3d(kernel_size = 3, stride = 2, padding = 1)

        self.inv_branch_left = nn.Sequential(
                                                AEConv3d(64, 32, kernel_size = 3, stride = 1, padding = 1),
                                                AEConv3d(32, 48, kernel_size = 1, stride = 1, padding = 0)
                                            )

        self.inv_branch_right = nn.Sequential(
                                                AEConv3d(64, 32, kernel_size = 1, stride = 1, padding = 0),
                                                AEConv3d(32, 32, kernel_size = (7,1,1), stride = 1, padding = (3, 0, 0)),
                                                AEConv3d(32, 32, kernel_size = (1,7,1), stride = 1, padding = (0, 3, 0)),
                                                AEConv3d(32, 32, kernel_size = (1,1,7), stride = 1, padding = (0, 0, 3)),
                                                AEConv3d(32, 48, kernel_size = 3, stride = 1, padding = 1)   
                                            )

        self.inv_downsample_left = nn.MaxUnpool3d(kernel_size = 3, stride = 2, padding = 1)
        self.inv_downsample_right = AEConv3d(32, 16, kernel_size = 3, stride = 2, padding = 1)

        self.inv_stem = nn.Sequential(
                                        AEConv3d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                                        AEConv3d(16, 8, kernel_size = 3, stride = 1, padding = 1),
                                        AEConv3d(8, 1, kernel_size = 3, stride = 2, padding = 1)
                                    )

    def forward(self, x, idx1, idx2):
        left_branch, right_branch = torch.split(x, [128, 128], dim = 1)

        left_branch = self.inv_left(left_branch)
        right_branch = self.inv_right(right_branch, idx1)

        x = torch.mean(torch.stack([left_branch, right_branch]))

        left_branch, right_branch = torch.split(x, [64, 64], dim = 1)

        left_branch = self.inv_branch_left(left_branch)
        right_branch = self.inv_branch_right(right_branch)

        x = torch.mean(torch.stack([left_branch, right_branch]))

        left_branch, right_branch = torch.split(x, [16, 32], dim = 1)

        left_branch = self.inv_downsample_left(left, idx2)
        right_branch = self.inv_downsample_right(right)

        x = torch.mean(torch.stack([left_branch, right_branch]))

        x = self.inv_stem(x)

        return x

class InceptionReduce(nn.Module):
    def __init__(self, in_features):
        super(InceptionReduce, self).__init__()

        self.branch1 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1, return_indices = True)

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

        b1, idx = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b1, b2, b3], dim = 1)

        return x, idx

class AEInceptionReduce(nn.Module):
    def __init__(self, in_features, split_features):
        super(AEInceptionReduce, self).__init__()

        self.split_features = split_features

        self.inv_branch1 = nn.MaxUnpool3d(kernel_size = 2, stride = 2, padding = 0)

        self.inv_branch2 = nn.Sequential(
                                            AEConv3d(128, 128, kernel_size = 1, stride = 1, padding = 0),
                                            AEConv3d(128, in_features, kernel_size = 3, stride = 2, padding = 0)
                                        )

        self.inv_branch3 = nn.Sequential(
                                            AEConv3d(256, 256, kernel_size = 3, stride = 2, padding = 1),
                                            AEConv3d(256, 256, kernel_size = (7, 1, 1), stride = 1, padding = (3, 0, 0)),
                                            AEConv3d(256, 128, kernel_size = (1, 7, 1), stride = 1, padding = (0, 3, 0)),
                                            AEConv3d(128, 128, kernel_size = (1, 1, 7), stride = 1, padding = (0, 0, 3)),
                                            AEConv3d(128, in_features, kernel_size = 1, stride = 1, padding = 0)
                                        )

    def forward(self, x, idx):
        branch1, branch2, branch3 = torch.split(x, [self.split_features, 128, 256], dim = 1)
        print(branch2.shape)
        
        branch1 = self.inv_branch1(branch1, idx)
        branch2 = self.inv_branch2(branch2)
        branch3 = self.inv_branch3(branch3)

        print(branch2.shape)
        input()


  

        x = torch.mean(torch.stack([branch1, branch2, branch3]))

        # [-1, 896, 2, 4, 4] -> [-1, 512, 4, 8, 8]
        return x

class AEInceptionModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(AEInceptionModel, self).__init__()
        self.identifier = 'AEInceptionModel'
        self.dims = (in_depth, in_height, in_width)

        self.stem = InceptionStem()

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

        self.inv_reduce2 = AEInceptionReduce(512, 512)

        self.inv_incept10 = AEInceptionNode(512)
        self.inv_incept9 = AEInceptionNode(512)
        self.inv_incept8 = AEInceptionNode(512)
        self.inv_incept7 = AEInceptionNode(512)
        self.inv_incept6 = AEInceptionNode(512)
        self.inv_incept5 = AEInceptionNode(512)
        self.inv_incept4 = AEInceptionNode(512)

        self.inv_reduce1 = AEInceptionReduce(896, 512)

        self.inv_incept3 = AEInceptionNode(512)
        self.inv_incept2 = AEInceptionNode(512)
        self.inv_incept1 = AEInceptionNode(512)

        self.inv_stem = AEInceptionStem(256)


    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x, stem_idx1, stem_idx2 = self.stem(x)

        x, idx1 = self.incept1(x)
        x, idx2 = self.incept2(x)
        x, idx3 = self.incept3(x)

        x, reduce1_idx = self.reduce1(x)

        x, idx4 = self.incept4(x)
        x, idx5 = self.incept5(x)
        x, idx6 = self.incept6(x)
        x, idx7 = self.incept7(x)
        x, idx8 = self.incept8(x)
        x, idx9 = self.incept9(x)
        x, idx10 = self.incept10(x)

        x, reduce2_idx = self.reduce2(x)

        x = self.inv_reduce2(x, reduce2_idx)

        x = self.inv_incept10(x, idx10)
        x = self.inv_incept9(x, idx9)
        x = self.inv_incept8(x, idx8)
        x = self.inv_incept7(x, idx7)
        x = self.inv_incept6(x, idx6)
        x = self.inv_incept5(x, idx5)
        x = self.inv_incept4(x, idx4)

        x = self.inv_reduce1(x, reduce1_idx)

        x = self.inv_incept3(x, idx3)
        x = self.inv_incept2(x, idx2)
        x = self.inv_incept1(x, idx1)

        x = self.inv_stem(x, stem_idx1, stem_idx2)

        return x  