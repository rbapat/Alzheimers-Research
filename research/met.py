from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import multiprocessing
import torch.nn as nn
import torch
import shutil
import os

from research.datasets.scored_dataset import DataParser
from research.util.Grapher import TrainGrapher
from research.models.densenet import DenseNet

# model hyperparameters
NUM_EPOCHS = 1000
BATCH_SIZE = 5
TOLERANCE = (0.1,)

# weight/graphing parameters
GRAPH_METRICS = True
SAVE_FREQ = 10
GRAPH_FREQ = 10

# data shapes
DATA_DIM = (128, 128, 128)


def main():
    dataset = DataParser(DATA_DIM)
    num_outputs = dataset.get_num_outputs()

    # initialize the data loaders needed for training and validation
    train_loader = DataLoader(dataset.get_subset(0), batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(dataset.get_subset(1), batch_size = BATCH_SIZE, shuffle = True)
    loaders = [train_loader, val_loader]

    model = DenseNet(DATA_DIM, num_outputs, [6, 12, 24, 16], growth_rate = 12, theta = 1.0, drop_rate = 0.0).cuda()

    with torch.no_grad():
        if os.path.exists('pretrain.t7'):
            ckpt = torch.load('pretrain.t7')

            for name, param in ckpt['state_dict'].items():
                if name not in model.state_dict():
                    continue

                if model.state_dict()[name].shape != param.shape:
                    #print("Failed", name, model.state_dict()[name].shape, 'was not', param.shape)
                    print("Failed", name)
                    continue

                model.state_dict()[name].copy_(param)
                #model.state_dict()[name].requires_grad = False

            print("Pretrained Weights Loaded!")


    y_label, y_preds = [], []
    for (data, label) in val_loader:
        data, label = data.cuda(), label.float().cuda()

        model.train(False)
        preds = model(data)

        
        y_label += [l[0].item() for l in label]
        y_preds += [p[0].item() for p in preds]

    print('-------------------')
    print(y_label)
    print('-------------------')
    print(y_preds)
    print('-------------------')


 
if __name__ == '__main__':
    main()
