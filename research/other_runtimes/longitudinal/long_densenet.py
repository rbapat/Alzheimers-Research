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
                                        nn.BatchNorm3d(in_features),
                                        nn.ReLU(),
                                        nn.Conv3d(in_features, out_features, bias=True, **kwargs)
                                    )

    def forward(self, x):
        return self.model(x)

class DenseUnit(nn.Module):
    def __init__(self, in_features, growth_rate, drop_rate):
        super(DenseUnit, self).__init__()

        self.bottleneck = Conv3d(in_features, 4 * growth_rate, kernel_size = 1, stride = 1, padding = 0)
        self.conv2 = Conv3d(4 * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1)
        self.drop = nn.Dropout3d(drop_rate)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)

        return x

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

class TransitionBlock(nn.Module):
    def __init__(self, in_features, theta, drop_rate):
        super(TransitionBlock, self).__init__()

        self.conv = Conv3d(in_features, int(in_features * theta), kernel_size = 1, stride = 1, padding = 0)
        self.drop = nn.Dropout3d(drop_rate)
        self.pool = nn.AvgPool3d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.pool(x)

        return x

class DenseNet(nn.Module):
    def __init__(self, in_dims, out_features, channels, growth_rate, theta, drop_rate):
        super(DenseNet, self).__init__()

        # TODO: Make this cleaner
        self.identifier = 'DenseNet'
        #for param in (channels, growth_rate, theta, drop_rate):
        #    self.identifier = self.identifier + '_' + str(param)

        self.dims = in_dims

        compressed_size = int(growth_rate * theta)

        self.stem = nn.Sequential(
                                    Conv3d(1, 2 * growth_rate, kernel_size = 7, stride = 2, padding = 3),
                                    nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
                                )

        layers = [DenseBlock(2 * growth_rate, channels[0], growth_rate, drop_rate)] # 0 
        for idx, channel in enumerate(channels[1:]):
            layers.append(TransitionBlock(growth_rate, theta, drop_rate)) # 1 3 5
            layers.append(DenseBlock(compressed_size, channel, growth_rate, drop_rate)) # 2 4 6

        self.model = nn.Sequential(*layers)

        self.end_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout(0.7)
        
        
        conv_dims = (growth_rate * self.dims[0] * self.dims[1] * self.dims[2]) // 2**((len(channels)+1)*3)
        self.rnn = nn.LSTM(conv_dims, 256, 2, batch_first=True, dropout = 0.7) # TODO: Possibly Dropout?
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # TODO: Remove This (need to edit class_main/class_cam)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        bs, ts, h, w, d = x.shape
        x = x.view(bs * ts, 1, h, w, d)

        x = self.stem(x)

        x = self.model(x)

        #x = self.end_pool(x)
        x = x.view(bs, ts, -1)
        
        #x = self.drop(x)
        x, _ = self.rnn(x)
        #x = self.drop(x)
        #print(x)
        #print(x.shape)
        

        x = x.reshape(bs, -1)
        x = self.fc(x)
        #print(x)
        #print(x.shape)
        #print()
        x = self.relu(x)

        return x

    def init_optimizer(self):
        optim = torch.optim.SGD(self.parameters(), lr = 0.01, momentum = 0.9, weight_decay = .001, nesterov = True)
        #optim = torch.optim.Adam(self.parameters(), lr = 0.01);
        scheduler = None

        return optim, scheduler
