import torch.nn as nn
import numpy as np
import torch
import math

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
        #x = self.drop(x)
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
        #self.fc = nn.Sequential(nn.Linear(2 * 3 * 2 * growth_rate, 1000), nn.ReLU(), nn.Linear(1000, out_features)) # nn.Linear(growth_rate, out_features)
        self.fc = nn.Linear(growth_rate, out_features)
        self.drop = nn.Dropout(0.4)
        
        # Weight initialization, not sure if I actually need this
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)    
            elif isinstance(m, (nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
                print(f"Copied {name}")

                if requires_grad is not None:
                    self.state_dict()[name].requires_grad = requires_grad

            print("Pretrained Weights Loaded!")

    # Just gets the output of the final convolutional layers
    def features(self, x):
        x = x.view(-1, 1,  *self.dims)

        x = self.stem(x)

        x = self.model(x)

        x = self.end_pool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x, clin_vars):
        x = x.view(-1, 1,  *self.dims)

        x = self.stem(x)

        x = self.model(x)

        x = self.end_pool(x)
        x = torch.flatten(x, 1)
        # x = self.drop(x)
        x = self.fc(x)

        return x

# LSTM model that I'm using right now, relatively simple and can definitely be improved/tuned further
class LSTMNet(nn.Module):
    def __init__(self, conv_features = 16, lstm_hidden = 256, lstm_layers = 2):
        super(LSTMNet, self).__init__()

        conv_features += 4
        self.lstm = nn.LSTM(input_size = conv_features, hidden_size = lstm_hidden, num_layers = lstm_layers, bias = True, batch_first = True, dropout = 0.1)

        self.predictor = nn.Sequential(
                                        nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(),
                                        nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(),
                                        nn.Linear(lstm_hidden // 4, 2)
                                    )

    def forward(self, x, clin_vars):
        x = torch.cat([x, clin_vars], dim = 2)
        x, (h_n, c_n) = self.lstm(x) # num_layers, batch_size, lstm_hidden
            
        x = self.predictor(x[:, -1, :]) 
        return x