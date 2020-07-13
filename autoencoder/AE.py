from torch.utils.data import DataLoader
from dataset import DataParser
from util import TrainGrapher
import torch.optim as optim
from AEModel import *
import torch.nn as nn
import torch
import os

# model hyperparameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
BATCH_SIZE = 1
WEIGHT_DECAY = 0.00001

# weight/graphing parameters
LOAD_WEIGHT = True
SAVE_FREQ = 50
SAVE_MODEL = True
GRAPH_FREQ = 10
GRAPH_METRICS = True
EARLY_STOP_THRESH = 90

# data shapes
DATA_DIM = (128, 128, 64)
NUM_OUTPUTS = 2

def main():
    dataset = DataParser("dataset.csv", DATA_DIM, NUM_OUTPUTS, splits = [0.6, 0.2, 0.2])

    # initialize the data loaders needed for training and validation
    # TODO: increase validation/test batch size to something nicer than 1
    train_loader = DataLoader(dataset.get_loader(0), batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(dataset.get_loader(1), batch_size = 1, shuffle = True)
    test_loader = DataLoader(dataset.get_loader(2), batch_size = 1, shuffle = True)
    loaders = [train_loader, val_loader, test_loader]

    # initialize matplotlib graph and get references to lists to be graphed
    grapher = TrainGrapher(GRAPH_METRICS, "Loss")
    losses = grapher.add_lines("Loss", "Train Loss", "Validation Loss")

    # initialize model, loss function, and optimizer
    model = AEInceptionModel(*DATA_DIM, NUM_OUTPUTS).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)


    # if checkpoint directory doesnt exist, create one if needed
    if SAVE_MODEL and not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # run main training loop. Validation set is tested after every training iteration
    exit_early, stop_early = False, False
    for epoch in range(1, NUM_EPOCHS + 1):
        for phase in range(len(loaders)): # phase: 0 = train, 1 = val, 2 = test
            running_loss, running_correct = 0.0, 0.0

            # dont use test set unless we are in early stopping phase
            if not stop_early and phase == 2:
                continue

            for (data, label) in loaders[phase]:
                # convert data to cuda because model is cuda
                data = data.cuda()

                # eval mode changes behavior of dropout and batch norm for validation
                if phase == 0:
                    model.train()
                elif phase == 1:
                    model.eval()

                decoded = model(data)

                loss = criterion(decoded, data)
                optimizer.zero_grad()

                # backprop if in training phase
                if phase == 0:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * len(data)

            # get metrics over entire dataset
            true_loss = running_loss / len(dataset.get_loader(phase))

            if phase == 0:
                print("Epoch %d/%d, train loss: %.4f" % (epoch, NUM_EPOCHS, true_loss), end = "") 
            elif phase == 1:
                print(", val loss: %.4f" % true_loss)
            elif phase == 2:
                print("Model stopping early with a loss of %.4f" % true_loss)
                exit_early = True
                break

            # add metrics to list to be graphed
            if phase < 2:
                losses[phase].append(true_loss)

        # update graph with new data every GRAPH_FREQ epochs
        if epoch % GRAPH_FREQ == 0:
            grapher.update() 

        # output model weights to checkpoints directory if specified
        if exit_early or (SAVE_MODEL and epoch % SAVE_FREQ == 0):
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save model with early_stop_ prefix if needed
            prefix = ""
            if exit_early:
                prefix = "early_stop_"

            path = os.path.join('checkpoints', '%s_%sepoch_%d.t7' % (model.identifier, prefix, epoch))
            torch.save(state, path)

        if exit_early:
            break

    grapher.show()

if __name__ == '__main__':
    main()