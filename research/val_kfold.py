from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import multiprocessing
from datetime import datetime
import pandas as pd
import torch.nn as nn
import torch
import shutil
import os

from research.datasets.scored_dataset import DataParser
from research.datasets.base_dataset import TorchLoader
from research.util.Grapher import TrainGrapher
from research.models.densenet import DenseNet
from sklearn.model_selection import StratifiedKFold

# model hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 1
TOLERANCE = (0.1,)

# weight/graphing parameters
GRAPH_METRICS = False
SAVE_FREQ = 1
GRAPH_FREQ = 10

# data shapes
DATA_DIM = (128, 128, 128)

# load weights from a given file into model
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
    print("Initializing Data Loader")
    test_loader = DataLoader(dataset.get_subset(1), batch_size = BATCH_SIZE)

    # create dictionary mapping file path to diagnosis
    dx_dict = {}
    df = pd.read_csv('ADNIMERGE.csv', low_memory = False)
    for (root, dirs, files) in os.walk(os.path.join('ADNI', 'AllData_FSL')):
        for file in files:
            if file[-7:] == '.nii.gz':
                try:
                    image_id = int(file[file.rindex('_') + 2:-7])
                    dx_dict[os.path.join(root, file)] = ['CN', 'MCI', 'Dementia'].index(df[df['IMAGEUID'] == image_id]["DX"].values[0])
                except Exception as e:
                    print(e)
    
    # get diagnosis of all the data in validation set
    val_classes = []
    for path, score in dataset.get_subset_array(1):
        val_classes.append(dx_dict[path])
    

    print("Initializing Model")
    model = DenseNet(DATA_DIM, num_outputs, [6, 12, 32, 24], growth_rate = 24, theta = 1.0, drop_rate = 0.0).cuda()
    load_weights(model, 'pretrain.t7')
    criterion = nn.MSELoss()
    optimizer, scheduler = model.init_optimizer()    #grapher = TrainGrapher(GRAPH_METRICS, "Accuracy", "Loss")
    #accuracy = grapher.add_lines("Accuracy", 'lower left', "Train Accuracy", "Validation Accuracy")
    #losses = grapher.add_lines("Loss", 'upper right', "Train Loss", "Validation Loss")

    for idx, (data, label) in enumerate(test_loader):
        data, label = data.cuda(), label.float().cuda()
        model.train(False)

        preds = model(data)

        print(preds, label, val_classes[idx])
if __name__ == '__main__':
    main()
