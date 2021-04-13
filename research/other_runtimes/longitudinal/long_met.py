from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import multiprocessing
from datetime import datetime
import torch.nn as nn
import torch
import shutil
import os

from research.datasets.long_dataset import DataParser
from research.util.Grapher import TrainGrapher
from research.models.long_densenet import DenseNet

# model hyperparameters
NUM_EPOCHS = 1000
BATCH_SIZE = 1
TOLERANCE = (0.1,)

# weight/graphing parameters
GRAPH_METRICS = False
SAVE_FREQ = 1
GRAPH_FREQ = 10

# data shapes
DATA_DIM = (128, 128, 128)

def get_rsquared(x_list, y_list):
    corr_mat = np.corrcoef(x_list, y_list)
    corr_xy = corr_mat[0,1]
    r_sq = corr_xy**2

    return r_sq

def save_model(model, optimizer, epoch):
    state = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }

    path = os.path.join('checkpoints', '%s_epoch_%d.t7' % (model.identifier, epoch))
    torch.save(state, path)

def load_weights(model, weight_file, requires_grad = None):
    with torch.no_grad():
        if not os.path.exists(weight_file):
            print("Weight file %s not found" % weight_file)
            return

        ckpt = torch.load(weight_file)
        for name, param in ckpt['state_dict'].items():
            if name not in model.state_dict():
                continue

            if model.state_dict()[name].shape != param.shape:
                print("Failed", name, model.state_dict()[name].shape, 'was not', param.shape)
                continue

            model.state_dict()[name].copy_(param)

            if requires_grad is not None:
                model.state_dict()[name].requires_grad = requires_grad

        print("Pretrained Weights Loaded!")

def main():
    dataset = DataParser(DATA_DIM)
    num_outputs = dataset.get_num_outputs()

    # initialize the data loaders needed for training and validation
    train_loader = DataLoader(dataset.get_subset(0), batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(dataset.get_subset(1), batch_size = BATCH_SIZE, shuffle = True)
    model = DenseNet(DATA_DIM, num_outputs, [6, 12, 32, 24], growth_rate = 24, theta = 0.5, drop_rate = 0.0).cuda()

    load_weights(model, 'pretrain.t7')

    for (data, label) in val_loader:
        data, label = data.cuda(), label.type(torch.FloatTensor).cuda()

        model.train(False)
        preds = model(data)
        print("Predicted:", preds)
        print("Label:", label)


if __name__ == '__main__':
    main()
