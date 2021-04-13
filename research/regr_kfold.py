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
BATCH_SIZE = 5
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

# save model state to file
def save_model(model, optimizer, epoch, fold):
    state = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }

    path = os.path.join('checkpoints', '%s_epoch_%d_fold_%d.t7' % (model.identifier, epoch, fold))
    torch.save(state, path)

# load model weights from file into network
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
    #test_loader = DataLoader(dataset.get_subset(1), batch_size = BATCH_SIZE, shuffle = True)
    train_dataset = dataset.get_subset_array(0)
    
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
    train_classes = []
    for path, score in train_dataset:
        train_classes.append(dx_dict[path])

    print(train_classes)
    print(len(train_classes))

    folds = StratifiedKFold(n_splits = 5)

    #grapher = TrainGrapher(GRAPH_METRICS, "Accuracy", "Loss")
    #accuracy = grapher.add_lines("Accuracy", 'lower left', "Train Accuracy", "Validation Accuracy")
    #losses = grapher.add_lines("Loss", 'upper right', "Train Loss", "Validation Loss")

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    for fold_idx, (train_index, val_index) in enumerate(folds.split(train_dataset, train_classes)):
        print("Train Index:")
        print(train_index)
        print("Val Index:")
        print(val_index)
        print("\n\nEvaluating Fold", fold_idx)

        # initialize model, loss function, and optimizer
        model = DenseNet(DATA_DIM, num_outputs, [6, 12, 32, 24], growth_rate = 24, theta = 1.0, drop_rate = 0.0).cuda()
        load_weights(model, 'pretrain_%d.t7' % fold_idx)
        criterion = nn.MSELoss()
        optimizer, scheduler = model.init_optimizer()
        print("Model Initialized")
        print("Train:", len(train_index))
        print("Val:", len(val_index))
        
        # create data loaders for this fold
        train_loader = DataLoader(TorchLoader(train_dataset[train_index], DATA_DIM), batch_size = BATCH_SIZE, shuffle = True)
        val_loader = DataLoader(TorchLoader(train_dataset[val_index], DATA_DIM), batch_size = BATCH_SIZE, shuffle = True)
        loaders = [train_loader, val_loader]
        print("Loaders Initialized")

        for epoch in range(1, NUM_EPOCHS + 1):
            x_val, y_val = [], []
            for phase in range(len(loaders)): # phase: 0 = train, 1 = val
                running_loss, running_correct = 0.0, 0.0
                for (data, label) in loaders[phase]:
                    # convert data to cuda because model is cuda
                    data, label = data.cuda(), label.float().cuda()
                    
                    # eval mode changes behavior of dropout and batch norm for validation
                    model.train(phase == 0)

                    preds = model(data)
                    loss = criterion(preds, label)

                    optimizer.zero_grad()

                    # backprop if in training phase
                    if phase == 0:
                        loss.backward()
                        optimizer.step()

                        if scheduler is not None:
                            scheduler.step()
                    elif phase == 1:
                        for y_hat, y in zip(preds, label):
                            x_val.append(y[0].item())
                            y_val.append(y_hat[0].item())

                    running_loss += (loss.item() * len(data))

                    # maybe abstract this to the dataset?
                    difference = torch.abs(preds - label)
                    running_correct += sum([(difference[:, i] < x).sum().item() for i,x in enumerate(TOLERANCE)]) / float(num_outputs)

                if phase == 0:
                    length = len(train_index)
                else:
                    length = len(val_index)

                # get metrics over entire dataset
                true_accuracy = 100 * running_correct / length
                true_loss = running_loss / length

                if phase == 0:
                    print("[%s] " % datetime.now().strftime("%H:%M:%S"), end="")
                    print("Epoch %d/%d, train accuracy: %.2f, train loss: %.4f" % (epoch, NUM_EPOCHS, true_accuracy, true_loss), end ="") 
                elif phase == 1:
                    r_sq = get_rsquared(x_val, y_val)
                    print(", val accuracy: %.2f, val loss: %.4f, r squared: %.4f, min: %.4f, max: %.4f" % (true_accuracy, true_loss, r_sq, np.min(y_val), np.max(y_val)))

                # add metrics to list to be graphed
                #if phase < 2:
                #    accuracy[phase].append(true_accuracy)
                #    losses[phase].append(true_loss)

            # update graph with new data points
            #if epoch % GRAPH_FREQ == 0:
            #    grapher.update() 

            # output model weights to checkpoints directory if specified
            if epoch % SAVE_FREQ == 0:
                save_model(model, optimizer, epoch, fold_idx)

    #grapher.show()

if __name__ == '__main__':
    main()
