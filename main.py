from torch.utils.data import DataLoader
from dataset import DataParser
from util import TrainGrapher
import torch.optim as optim
from model import ADModel, ExperimentalModel
import torch.nn as nn
import torch
import os

# model hyperparameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
BATCH_SIZE = 10
WEIGHT_DECAY = 0.00001

# weight/graphing parameters
LOAD_WEIGHT = True
SAVE_FREQ = 100
SAVE_MODEL = True
GRAPH_FREQ = 10
GRAPH_METRICS = True

# data shapes
DATA_DIM = (128, 128, 64)
NUM_OUTPUTS = 2

def main():
    dataset = DataParser("dataset.csv", DATA_DIM, NUM_OUTPUTS)

    # initialize the data loaders needed for training and validation
    train_loader = DataLoader(dataset.get_loader(0), batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(dataset.get_loader(1), batch_size = 1, shuffle = True)
    loaders = [train_loader, val_loader]

    # initialize matplotlib graph and get references to lists to be graphed
    grapher = TrainGrapher(GRAPH_METRICS, "Accuracy", "Loss")
    accuracy = grapher.add_lines("Accuracy", "Train Accuracy", "Validation Accuracy")
    losses = grapher.add_lines("Loss", "Train Loss", "Validation Loss")

    # initialize model, loss function, and optimizer
    model = ExperimentalModel(*DATA_DIM, NUM_OUTPUTS).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    # load the model weights from disk if it exists
    if LOAD_WEIGHT and os.path.exists('optimal.t7'):
        ckpt = torch.load('optimal.t7')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # load weights from this model to only the first few layers
    if os.path.exists('pretrain.t7') and model.identifier == 'Baseline':
        #state_dict = torch.load('pretrain.t7')['state_dict']
        with torch.no_grad():
            ckpt = torch.load('pretrain.t7')
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

            print("Pretrained Weights Loaded!")

    # if checkpoint directory doesnt exist, create one if needed
    if SAVE_MODEL and not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # run main training loop. Validation set is tested after every training iteration
    for epoch in range(NUM_EPOCHS):
        for phase in range(2): # phase: 0 = train, 1 = val
            running_loss, running_correct = 0.0, 0.0
            for (data, label) in loaders[phase]:
                # convert data to cuda because model is cuda
                data, label = data.cuda(), label.type(torch.LongTensor).cuda()

                probs = model(data)

                # get class predictions
                label = torch.argmax(label, dim = 1)
                preds = torch.argmax(probs, dim = 1)

                loss = criterion(probs, label)
                optimizer.zero_grad()

                # backprop if in training phase
                if phase == 0:
                    loss.backward()
                    optimizer.step()

                # TODO: Loss calculation might be incorrect, check it
                running_loss += loss.item()
                running_correct += (preds == label).sum().item()

            # get metrics over entire dataset
            true_accuracy = 100 * running_correct / len(dataset.get_loader(phase))
            true_loss = running_loss / len(dataset.get_loader(phase))

            if phase == 0:
                print("Epoch %d/%d, train accuracy: %.2f" % (epoch + 1, NUM_EPOCHS, true_accuracy), end ="") 
            if phase == 1:
                print(", val accuracy: %.2f, val loss: %.4f" % (true_accuracy, true_loss))

            # add metrics to list to be graphed
            accuracy[phase].append(true_accuracy)
            losses[phase].append(true_loss)

        # update graph with new data every GRAPH_FREQ epochs
        if epoch % GRAPH_FREQ == 0:
            grapher.update()

        # output model weights to checkpoints directory if specified
        if SAVE_MODEL and epoch % SAVE_FREQ == 0:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            path = os.path.join('checkpoints', 'epoch_%d.t7' % epoch)
            torch.save(state, path)


    grapher.show()

if __name__ == '__main__':
    main()