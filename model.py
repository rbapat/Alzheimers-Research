import torch.nn as nn
import numpy as np
import torch

class ADModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(ADModel, self).__init__()
        self.version = '1.3'
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
                                nn.Dropout3d(0.1)
                            )

    def pretrain_weights(self, state_dict):
        self.begin_norm.weight.copy_(state_dict['begin_norm.weight'])
        self.begin_norm.bias.copy_(state_dict['begin_norm.bias'])

        self.block1[0].weight.copy_(state_dict['block1.0.weight'])
        self.block1[0].bias.copy_(state_dict['block1.0.bias'])
        self.block1[1].weight.copy_(state_dict['block1.1.weight'])
        self.block1[1].bias.copy_(state_dict['block1.1.bias'])

        self.block2[0].weight.copy_(state_dict['block2.0.weight'])
        self.block2[0].bias.copy_(state_dict['block2.0.bias'])
        self.block2[1].weight.copy_(state_dict['block2.1.weight'])
        self.block2[1].bias.copy_(state_dict['block2.1.bias'])

        self.block3[0].weight.copy_(state_dict['block3.0.weight'])
        self.block3[0].bias.copy_(state_dict['block3.0.bias'])
        self.block3[1].weight.copy_(state_dict['block3.1.weight'])
        self.block3[1].bias.copy_(state_dict['block3.1.bias'])

        self.block4[0].weight.copy_(state_dict['block4.0.weight'])
        self.block4[0].bias.copy_(state_dict['block4.0.bias'])
        self.block4[1].weight.copy_(state_dict['block4.1.weight'])
        self.block4[1].bias.copy_(state_dict['block4.1.bias'])

        self.block5[0].weight.copy_(state_dict['block5.0.weight'])
        self.block5[0].bias.copy_(state_dict['block5.0.bias'])
        self.block5[1].weight.copy_(state_dict['block5.1.weight'])
        self.block5[1].bias.copy_(state_dict['block5.1.bias'])

        self.conv1.weight.copy_(state_dict['conv1.weight'])
        self.conv1.bias.copy_(state_dict['conv1.bias'])

        self.end_norm.weight.copy_(state_dict['end_norm.weight'])
        self.end_norm.bias.copy_(state_dict['end_norm.bias'])

        self.linear.weight.copy_(state_dict['linear.weight'])
        self.linear.bias.copy_(state_dict['linear.bias'])


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