import torch.nn as nn
import numpy as np
import torch

class InceptionNodeV1(nn.Module):
    def __init__(self, in_dim, d1x1, r3x3, d3x3, r5x5, d5x5, dproj):
        super(InceptionNodeV1, self).__init__()

        self.branch1 = nn.Conv3d(in_dim, d1x1, kernel_size = 1, stride = 1, padding = 0)

        self.branch2 = nn.Sequential(
                                        nn.Conv3d(in_dim, r3x3, kernel_size = 1, stride = 1, padding = 0),
                                        nn.Conv3d(r3x3, d3x3, kernel_size = 3, stride = 1, padding = 1)
                                    )

        self.branch3 = nn.Sequential(
                                        nn.Conv3d(in_dim, r5x5, kernel_size = 1, stride = 1, padding = 0),
                                        nn.Conv3d(r5x5, d5x5, kernel_size = 5, stride = 1, padding = 2)
                                    )

        self.branch4 = nn.Sequential(
                                        nn.MaxPool3d(kernel_size = 3, stride = 1, padding = 1),
                                        nn.Conv3d(in_dim, dproj, kernel_size = 1, stride = 1, padding = 0)
                                    )
        self.drop = nn.Dropout3d(0.1)


    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        cat = torch.cat([b1, b2, b3, b4], dim = 1)

        x = self.drop(cat)
        return x



class InceptionModelV1(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(InceptionModelV1, self).__init__()
        self.identifier = 'InceptionModel'
        self.dims = (in_depth, in_height, in_width)

        self.intro_block = nn.Sequential(
                                            nn.Conv3d(1, 64, kernel_size = 7, stride = 2, padding  = 3),
                                            nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),
                                            nn.BatchNorm3d(64),
                                            nn.Conv3d(64, 64, kernel_size = 1, stride = 1, padding = 0),
                                            nn.Conv3d(64, 192, kernel_size = 3, stride = 1, padding = 1),
                                            nn.BatchNorm3d(192),
                                            nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
                                        )

                            #in, d1x1, r3x3, d3x3, r5x5, d5x5, dproj 
        #self.incept1 = InceptionNodeV1(192, 64, 96, 128, 16, 32, 32)
        #self.incept2 = InceptionNodeV1(256, 128, 128, 192, 32, 96, 64)

        self.pool1 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        
        #480 
        self.incept3 = InceptionNodeV1(192, 192, 96, 208, 16, 48, 64)
        self.incept4 = InceptionNodeV1(512, 160, 112, 224, 24, 64, 64)
        self.incept5 = InceptionNodeV1(512, 128, 128, 256, 24, 64, 64)
        #self.incept6 = InceptionNodeV1(512, 112, 144, 288, 32, 64, 64)
        #self.incept7 = InceptionNodeV1(528, 256, 160, 320, 32, 128, 128)

        self.pool2 = nn.AvgPool3d(kernel_size = 2, stride = 2, padding = 0)
        
        #832
        self.linear_dims = 512 * in_width * in_height * in_depth // (2**5)**3

        self.drop = nn.Dropout(0.4)
        self.linear = nn.Linear(self.linear_dims, 1024)
        self.softmax = nn.Softmax(dim = 1)

        self.classifier = nn.Linear(1024, out_features)

    def forward(self, x):
        x = x.view(-1, 1, *self.dims)

        x = self.intro_block(x)

        #x = self.incept1(x)
        #x = self.incept2(x)

        x = self.pool1(x)

        x = self.incept3(x)
        x = self.incept4(x)
        x = self.incept5(x)
        #x = self.incept6(x)
        #x = self.incept7(x)

        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.drop(x)
        x = self.linear(x)

        x = self.classifier(x)
    
        return x