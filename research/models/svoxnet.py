import torch.nn as nn
import numpy as np
import torch

class Conv3d(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(Conv3d, self).__init__()
        self.model = nn.Sequential(
                                        nn.Conv3d(in_features, out_features, bias=False, **kwargs),
                                        #nn.BatchNorm3d(out_features),
                                        nn.ReLU()
                                    )

    def forward(self, x):
        return self.model(x)    

class SVoxNet(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(SVoxNet, self).__init__()
        self.identifier = 'SVoxNet'
        self.dims = (in_depth, in_height, in_width)     

        self.conv1 = nn.Sequential(
                                    Conv3d(1, 16, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0),
                                    nn.BatchNorm3d(16)
                                )

        self.conv2 = nn.Sequential(
                                    Conv3d(16, 32, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(32, 32, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 4, stride = 4, padding = 0),
                                    nn.BatchNorm3d(32)
                                )

        self.conv3 = nn.Sequential(
                                    Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 4, stride = 4, padding = 0),
                                    nn.BatchNorm3d(64)
                                )

        self.linear1 = nn.Linear((64 * in_height * in_width * in_depth) // (2**5)**3, 1000)
        self.drop = nn.Dropout(0.5)

        self.linear2 = nn.Linear(1000, 200)

        self.classifier = nn.Linear(200, out_features)
       
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.classifier(x)

        return x