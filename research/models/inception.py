import torch.nn as nn
import skimage.transform
import numpy as np
import torch

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

class VFactorizedConv(nn.Module):
    def __init__(self, in_features, out_features, ksize):
        super(VFactorizedConv, self).__init__()

        self.conv = nn.Sequential(
                                    Conv3d(in_features, in_features, kernel_size = (ksize, 1, 1), stride = 1, padding = (ksize // 2, 0, 0)),
                                    Conv3d(in_features, in_features, kernel_size = (1, ksize, 1), stride = 1, padding = (0, ksize // 2, 0)),
                                    Conv3d(in_features, out_features, kernel_size = (1, 1, ksize), stride = 1, padding = (0, 0, ksize // 2)),
                                )

    def forward(self, x):
        x = self.conv(x)

        return x

class HFactorizedConv(nn.Module):
    def __init__(self, in_features, out_features, ksize):
        super(HFactorizedConv, self).__init__()

        self.conv1 = Conv3d(in_features, out_features, kernel_size = (ksize, 1, 1), stride = 1, padding = (ksize // 2, 0, 0))
        self.conv2 = Conv3d(in_features, out_features, kernel_size = (1, ksize, 1), stride = 1, padding = (0, ksize // 2, 0))
        self.conv3 = Conv3d(in_features, out_features, kernel_size = (1, 1, ksize), stride = 1, padding = (0, 0, ksize // 2))



    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)

        return c1, c2, c3

# 160
class InceptionNodeA(nn.Module):
    def __init__(self, in_features):
        super(InceptionNodeA, self).__init__()

        self.channel1 = nn.Sequential(
                                        Conv3d(in_features, 32, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(64, 64, kernel_size = 3, stride = 1, padding = 1)
                                    )

        self.channel2 = nn.Sequential(
                                        Conv3d(in_features, 16, kernel_size = 1, stride = 1, padding = 0),
                                        Conv3d(16, 32, kernel_size = 3, stride = 1, padding = 1)
                                    )

        self.channel3 = nn.Sequential(
                                        nn.AvgPool3d(kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(in_features, 32, kernel_size = 3, stride = 1, padding = 1)
                                    )

        self.channel4 = Conv3d(in_features, 32, kernel_size = 3, stride = 1, padding = 1)


    def forward(self, x):
        ch1 = self.channel1(x)
        ch2 = self.channel2(x)
        ch3 = self.channel3(x)
        ch4 = self.channel4(x)

        x = torch.cat([ch1, ch2, ch3, ch4], dim = 1)

        return x

# 192 + in
class InceptionReduceA(nn.Module):
    def __init__(self, in_features):
        super(InceptionReduceA, self).__init__()

        self.channel1 = nn.Sequential(
                                        Conv3d(in_features, 32, kernel_size =  1, stride = 1, padding = 0),
                                        Conv3d(32, 64, kernel_size =  3, stride = 1, padding = 1),
                                        Conv3d(64, 64, kernel_size =  3, stride = 2, padding = 1)
                                    )

        self.channel2 = Conv3d(in_features, 128, kernel_size =  3, stride = 2, padding = 1)

        self.channel3 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        ch1 = self.channel1(x)
        ch2 = self.channel2(x)
        ch3 = self.channel3(x)

        x = torch.cat([ch1, ch2, ch3], dim = 1)

        return x

# 512
class InceptionNodeB(nn.Module):
    def __init__(self, in_features):
        super(InceptionNodeB, self).__init__()

        self.channel1 = nn.Sequential(
                                        Conv3d(in_features, 64, kernel_size = 1, stride = 1, padding = 0),
                                        VFactorizedConv(64, 64, 7),
                                        VFactorizedConv(64, 128, 7),
                                    )

        self.channel2 = nn.Sequential(
                                        Conv3d(in_features, 64, kernel_size = 1, stride = 1, padding = 0),
                                        VFactorizedConv(64, 128, 7),
                                    )

        self.channel3 = nn.Sequential(
                                        nn.AvgPool3d(kernel_size = 3, stride = 1, padding = 1),
                                        Conv3d(in_features, 128, kernel_size = 1, stride = 1, padding = 0)
                                    )

        self.channel4 = Conv3d(in_features, 128, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        ch1 = self.channel1(x)
        ch2 = self.channel2(x)
        ch3 = self.channel3(x)
        ch4 = self.channel4(x)

        x = torch.cat([ch1, ch2, ch3, ch4], dim = 1)

        return x

# 384 + in
class InceptionReduceB(nn.Module):
    def __init__(self, in_features):
        super(InceptionReduceB, self).__init__()

        self.channel1 = nn.Sequential(
                                        Conv3d(in_features, 128, kernel_size =  1, stride = 1, padding = 0),
                                        Conv3d(128, 256, kernel_size =  3, stride = 2, padding = 1)
                                    )

        self.channel2 = nn.Sequential(
                                Conv3d(in_features, 128, kernel_size =  1, stride = 1, padding = 0),
                                VFactorizedConv(128, 128, 7),
                                Conv3d(128, 128, kernel_size = 3, stride = 2, padding = 1)
                            )

        self.channel3 = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        ch1 = self.channel1(x)
        ch2 = self.channel2(x)
        ch3 = self.channel3(x)

        x = torch.cat([ch1, ch2, ch3], dim = 1)

        return x

# 1440
class InceptionNodeC(nn.Module):
    def __init__(self, in_features):
        super(InceptionNodeC, self).__init__()

        self.channel1_stem = nn.Sequential(
                                            Conv3d(in_features, 192, kernel_size = 1, stride = 1, padding = 0),
                                            Conv3d(192, 192, kernel_size = 3, stride = 1, padding = 1)
                                        )
        self.channel1_end = HFactorizedConv(192, 192, 3)

        self.channel2_stem = Conv3d(in_features, 192, kernel_size = 1, stride = 1, padding = 0)
        self.channel2_end = HFactorizedConv(192, 192, 3)

        self.channel3 = nn.Sequential(
                                            nn.AvgPool3d(kernel_size = 3, stride = 1, padding = 1),
                                            Conv3d(in_features, 96, kernel_size = 1, stride = 1, padding = 0)
                                        )

        self.channel4 = Conv3d(in_features, 192, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        ch1 = self.channel1_stem(x)
        ch1_a, ch1_b, ch1_c = self.channel1_end(ch1)

        ch2 = self.channel2_stem(x)
        ch2_a, ch2_b, ch2_c = self.channel2_end(ch2)

        ch3 = self.channel3(x)

        ch4 = self.channel4(x)

        x = torch.cat([ch1_a, ch1_b, ch1_c, ch2_a, ch2_b, ch2_c, ch3, ch4], dim = 1)
        return x

class InceptionModel(nn.Module):
    def __init__(self, in_height, in_width, in_depth, out_features):
        super(InceptionModel, self).__init__()
        self.identifier = 'InceptionV3x'
        self.dims = (in_depth, in_height, in_width)
        
        self.stem = nn.Sequential(
                                    Conv3d(1, 16, kernel_size = 3, stride = 2, padding = 1),
                                    Conv3d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                                    Conv3d(16, 32, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),
                                    Conv3d(32, 48, kernel_size = 1, stride = 1, padding = 0),
                                    Conv3d(48, 64, kernel_size = 3, stride = 1, padding = 1),
                                    nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
                                )

        self.inceptA1 = InceptionNodeA(64)
        self.inceptA2 = InceptionNodeA(160)
        self.inceptA3 = InceptionNodeA(160)

        self.reduceA = InceptionReduceA(160)

        self.inceptB1 = InceptionNodeB(352)
        self.inceptB2 = InceptionNodeB(512)
        self.inceptB3 = InceptionNodeB(512)
        #self.inceptB4 = InceptionNodeB(512)
        #self.inceptB5 = InceptionNodeB(512)

        self.reduceB = InceptionReduceB(512)

        self.inceptC1 = InceptionNodeC(896)
        #self.inceptC2 = InceptionNodeC(1440)

        self.end_pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc = nn.Linear(1440, out_features)
        self.drop = nn.Dropout(0.4)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        masks = []

        '''
        with torch.no_grad():
            for batch_item in x.cpu():
                smaller_mat = skimage.transform.resize(batch_item, (4, 8, 8), anti_aliasing = True, preserve_range = True, mode = 'edge')
                smaller_mat = smaller_mat - np.min(smaller_mat)
                smaller_mat = smaller_mat / np.max(smaller_mat)
                smaller_mat = np.uint8(255 * smaller_mat)
                smaller_mask = torch.Tensor(smaller_mat) < 100

                masks.append(smaller_mask)
        '''

        x = x.view(-1, 1, *self.dims)

        x = self.stem(x)

        x = self.inceptA1(x)
        x = self.inceptA2(x)
        x = self.inceptA3(x)

        x = self.reduceA(x)

        x = self.inceptB1(x)
        x = self.inceptB2(x)
        x = self.inceptB3(x)
        #x = self.inceptB4(x)
        #x = self.inceptB5(x)

        x = self.reduceB(x)

        x = self.inceptC1(x)
        #x = self.inceptC2(x)
        
        '''
        with torch.no_grad():
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j][masks[i]] = 0
        '''
        

        x = self.end_pool(x)

        x = torch.flatten(x, 1)

        x = self.drop(x)
        x = self.fc(x)

        return x