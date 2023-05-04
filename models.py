import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import math
import os

# Wrapper for convolution operations used in DenseNet
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

# DenseUnit that essentially the DenseNet convolution operation
class DenseUnit(nn.Module):
    def __init__(self, in_features, growth_rate, drop_rate):
        super(DenseUnit, self).__init__()

        self.bottleneck = Conv3d(in_features, 4 * growth_rate, kernel_size = 1, stride = 1, padding = 0)
        self.conv2 = Conv3d(4 * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1)
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
            layers.append(DenseUnit(in_features + i * growth_rate, growth_rate, drop_rate))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for idx, bottleneck in enumerate(self.layers):
            new_x = bottleneck(x)

            if idx < len(self.layers) - 1:
                x = torch.cat([new_x, x], dim = 1)
            else:
                x = new_x

        return x

# Transition block for dimensionality reduction between DenseBlocks
class TransitionBlock(nn.Module):
    def __init__(self, in_features, theta, drop_rate):
        super(TransitionBlock, self).__init__()


        self.conv = Conv3d(in_features, int(in_features * theta), kernel_size = 1, stride = 1, padding = 0)
        self.norm = nn.BatchNorm3d(int(in_features * theta))
        self.drop = nn.Dropout3d(drop_rate)
        self.pool = nn.AvgPool3d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.pool(x)

        return x

# Main DenseNet implementation I'm using right now, implementation was based on the official pytorch implementation
class DenseNet(nn.Module):
    def __init__(self, in_dims, out_features, channels, growth_rate, theta, drop_rate):
        super(DenseNet, self).__init__()

        self.dims = in_dims

        compressed_size = int(growth_rate * theta)

        self.stem = nn.Sequential(
                                    Conv3d(1, 2 * growth_rate, kernel_size = 7, stride = 2, padding = 3),
                                    nn.MaxPool3d(kernel_size = 2, stride = 4, padding = 0)
                                )

        layers = [DenseBlock(2 * growth_rate, channels[0], growth_rate, drop_rate)] # 0 
        for idx, channel in enumerate(channels[1:]):
            layers.append(TransitionBlock(growth_rate, theta, drop_rate)) # 1 3 5
            layers.append(DenseBlock(compressed_size, channel, growth_rate, drop_rate)) # 2 4 6

        self.model = nn.Sequential(*layers)

        self.end_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(growth_rate, out_features)
        

    # Reads previously saved model weights. This is more complicated than it needs to be but has more verbose errors just in case
    def load_weights(self, weight_file, requires_grad = None):
        with torch.no_grad():
            if not os.path.exists(weight_file):
                print("Weight file %s not found" % weight_file)
                return

            ckpt = torch.load(weight_file)
            for name, param in ckpt['state_dict'].items():
                if name not in self.state_dict():
                    print(f"Failed to load weight {name} from checkpoint, it doesn't exist in given model")

                if self.state_dict()[name].shape != param.shape:
                    print("Failed", name, self.state_dict()[name].shape, 'was not', param.shape)
                    continue

                self.state_dict()[name].copy_(param)
                #print(f"Copied {name}")

                if requires_grad is not None:
                    self.state_dict()[name].requires_grad = requires_grad

            print("Pretrained Weights Loaded!")

    def features(self, x):
        x = x.view(-1, 1, *self.dims)
        x = self.stem(x)
        x = self.model(x)
        x = x.view(len(x), -1)
        return x

    def forward(self, x, clin_vars):
        x = x.view(-1, 1,  *self.dims)

        x = self.stem(x)

        x = self.model(x)

        x = self.end_pool(x).view(len(clin_vars), -1)
        x = self.fc(x)

        return x

class MultiModalNet(nn.Module):
    def __init__(self, num_features, num_cv):
        super().__init__()
        
        
        self.model = nn.Sequential(
                                    nn.Conv1d(in_channels = num_features + num_cv, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = True),
                                    nn.ReLU(),
                                    nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = True),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size = 2, stride = 2, padding = 0),
                                    nn.Flatten(1),
                                    nn.Linear(256, 2)
                                )
       
        
        '''  
        self.model = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(3 * (num_features + num_cv), 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 2)
                                )
        '''

    def forward(self, x, cv):
        x = torch.cat([x, cv], dim = -1)
        x = x.transpose(1, 2)
        return self.model(x)

class ImageOnly(nn.Module):
    def __init__(self, num_features, num_cv):
        super().__init__()
        
        self.model = nn.Sequential(
                                    nn.Conv1d(in_channels = num_features, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = True),
                                    nn.ReLU(),
                                    nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = True),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size = 2, stride = 2, padding = 0),
                                    nn.Flatten(1),
                                    nn.Linear(256, 2)
                                )

    def forward(self, x, cv):
        x = x.transpose(1, 2)
        return self.model(x)


class CVOnly(nn.Module):
    def __init__(self, num_features, num_cv):
        super().__init__()
        
        self.model = nn.Sequential(
                                    nn.Conv1d(in_channels = num_cv, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = True),
                                    nn.ReLU(),
                                    nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = True),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size = 2, stride = 2, padding = 0),
                                    nn.Flatten(1),
                                    nn.Linear(256, 2)
                                )

    def forward(self, x, cv):
        cv = cv.transpose(1, 2)
        return self.model(cv)

# https://pytorch.org/vision/0.12/_modules/torchvision/models/vgg.html
class VGGNet(nn.Module):
    def __init__(self, in_dims, out_features):
        super().__init__()

        self.dims = in_dims

        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
        layers = []
        in_channels = 1

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool3d(kernel_size = 2, stride = 2))
            else:
                layers.append(nn.Conv3d(in_channels, v, kernel_size = 3, stride = 1, padding = 1))
                layers.append(nn.ReLU())
                in_channels = v

        self.model = nn.Sequential(*layers)
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool3d((7, 7, 7)),
            nn.Flatten(1),
            nn.Linear(512 * 7 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, out_features),
        )

    def forward(self, x, cv):
        x = x.view(-1, 1, *self.dims)

        x = self.model(x)
        x = self.predictor(x)

        return x

class DilationNet(nn.Module):
    def __init__(self, in_dims, out_features):
        super().__init__()

        self.dims = in_dims

        self.model = nn.Sequential(
                                    nn.Conv3d(1, 4, kernel_size = 3, stride = 1, padding = 3, dilation = 3),
                                    nn.BatchNorm3d(4),
                                    nn.LeakyReLU(),
                                    nn.Conv3d(4, 4, kernel_size = 3, stride = 1, padding = 3, dilation = 3),
                                    nn.BatchNorm3d(4),
                                    nn.LeakyReLU(),
                                    nn.MaxPool3d(kernel_size = 4, stride = 2),
                                    nn.Conv3d(4, 8, kernel_size = 3, stride = 1, padding = 3, dilation = 3),
                                    nn.BatchNorm3d(8),
                                    nn.LeakyReLU(),
                                    nn.Conv3d(8, 8, kernel_size = 3, stride = 1, padding = 3, dilation = 3),
                                    nn.BatchNorm3d(8),
                                    nn.LeakyReLU(),
                                    nn.MaxPool3d(kernel_size = 4, stride = 2),
                                    nn.Conv3d(8, 16, kernel_size = 3, stride = 1, padding = 3, dilation = 3),
                                    nn.BatchNorm3d(16),
                                    nn.LeakyReLU(),
                                    nn.Conv3d(16, 16, kernel_size = 3, stride = 1, padding = 3, dilation = 3),
                                    nn.BatchNorm3d(16),
                                    nn.LeakyReLU(),
                                    nn.MaxPool3d(kernel_size = 4, stride = 2),
                                    nn.Conv3d(16, 32, kernel_size = 3, stride = 1, padding = 3, dilation = 3),
                                    nn.BatchNorm3d(32),
                                    nn.LeakyReLU(),
                                    nn.Conv3d(32, 32, kernel_size = 3, stride = 1, padding = 3, dilation = 3),
                                    nn.BatchNorm3d(32),
                                    nn.LeakyReLU(),
                                    nn.MaxPool3d(kernel_size = 4, stride = 2),
                                    nn.Flatten(1),
                                    nn.Linear(9 * 9 * 11 * 32, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 2)
                                )

    def forward(self, x, cv):
        x = x.view(-1, 1, *self.dims)
        x = self.model(x)

        return x
