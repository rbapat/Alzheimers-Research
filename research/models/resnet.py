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
                                        nn.BatchNorm3d(out_features),
                                        nn.ReLU()
                                    )

    def forward(self, x):
        return self.model(x)



# TODO: Merge Bottleneck Block with BasicBlock
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_features, out_features, downsample):
        super(BottleneckBlock, self).__init__()

        # TODO: CLEAN THIS, IT IS HACKY AND UGLY :(
        if downsample:
            self.shortcut = Conv3d(in_features * self.expansion, out_features * self.expansion, kernel_size = 1, stride = 2, padding = 0)
            self.conv1 = Conv3d(in_features * self.expansion, out_features, kernel_size = 1, stride = 2, padding = 0)
        elif in_features != out_features * self.expansion:
            self.shortcut = Conv3d(in_features, out_features * self.expansion, kernel_size = 1, stride = 1, padding = 0)
            self.conv1 = Conv3d(in_features, out_features, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.shortcut = nn.Identity()
            self.conv1 = Conv3d(in_features, out_features, kernel_size = 1, stride = 1, padding = 0)
        
        self.conv2 = Conv3d(out_features, out_features, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = Conv3d(out_features, out_features * self.expansion, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += residual
        return x

# TODO: Merge Bottleneck Block with BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features, out_features, downsample):
        super(BasicBlock, self).__init__()

        if downsample:
            self.shortcut = Conv3d(in_features, out_features, kernel_size = 1, stride = 2, padding = 0)
            self.conv1 = Conv3d(in_features, out_features, kernel_size = 3, stride = 2, padding = 1)
        else:
            self.shortcut = nn.Identity()
            self.conv1 = Conv3d(in_features, out_features, kernel_size = 3, stride = 1, padding = 1)

        self.conv2 = Conv3d(out_features, out_features, kernel_size = 3, stride = 1, padding = 1)


    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x += residual
        return x



class ResidualBlock(nn.Module):
    def  __init__(self, block, num_blocks, in_features, downsample):
        super(ResidualBlock, self).__init__()

        layers = []
        for i in range(num_blocks):
            if downsample:
                layers.append(block(in_features, in_features * 2, downsample))
                downsample = False
                in_features *= 2
            elif i > 0:
                layers.append(block(in_features * block.expansion, in_features, downsample))
            else:
                layers.append(block(in_features, in_features, downsample))


        self.model = nn.Sequential(*layers)


    def forward(self, x):
        x = self.model(x)

        return x

class ResNet(nn.Module):
    def __init__(self, in_dims, out_features, channels, bottleneck):
        super(ResNet, self).__init__()
        self.identifier = 'ResNet'
        self.dims = in_dims

        self.stem = nn.Sequential(
                                    Conv3d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                                    nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
                                )

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        cur_features = 64
        layers = []
        for idx, channel in enumerate(channels):
            layers.append(ResidualBlock(block, channels[idx], cur_features, idx > 0))

            if idx > 0:
                cur_features *= 2

        self.model = nn.Sequential(*layers)
        self.end_pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc = nn.Linear(cur_features * block.expansion, out_features)
      
    def forward(self, x):
        x = x.view(-1, 1, *self.dims)
        x = self.stem(x)

        x = self.model(x)

        x = self.end_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def init_optimizer(self):
        optim =  torch.optim.SGD(self.parameters(), lr = 0.01, momentum = 0.9, weight_decay = .00001)
        scheduler = None

        return optim, scheduler
