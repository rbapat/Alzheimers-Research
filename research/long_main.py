from torch.utils.data import DataLoader
import torch.optim as optim
import multiprocessing
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
GRAPH_METRICS = True
SAVE_FREQ = 10
GRAPH_FREQ = 10

# data shapes
DATA_DIM = (128, 128, 128)


def main():
    dataset = DataParser(DATA_DIM)

    # initialize the data loaders needed for training and validation
    train_loader = DataLoader(dataset.get_subset(0), batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(dataset.get_subset(1), batch_size = BATCH_SIZE, shuffle = True)
    loaders = [train_loader, val_loader]

    grapher = TrainGrapher(GRAPH_METRICS, "Accuracy", "Loss")
    accuracy = grapher.add_lines("Accuracy", 'lower left', "Train Accuracy", "Validation Accuracy")
    losses = grapher.add_lines("Loss", 'upper right', "Train Loss", "Validation Loss")
        
    #model = DenseNet(DATA_DIM, num_outputs, [6, 12, 24, 16], drop_rate = 0.0).cuda()
    model = DenseNet(*DATA_DIM).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = model.init_optimizer()

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    '''
    with torch.no_grad():
        ckpt = torch.load('weights.t7')
        for name, param in ckpt['state_dict'].items():
            if name not in model.state_dict() or model.state_dict()[name].shape != param.shape:
                continue

            model.state_dict()[name].copy_(param)
            model.state_dict()[name].requires_grad = False

            print("Loaded", name)

        print("Pretrained Weights Loaded!")
    '''

    for epoch in range(1, NUM_EPOCHS + 1):
        for phase in range(len(loaders)): # phase: 0 = train, 1 = val, 2 = test
            running_loss, running_correct = 0.0, 0.0

            if phase == 2:
                continue

            for (data, label) in loaders[phase]:
                # convert data to cuda because model is cuda
                data, label = data.cuda(), label.type(torch.LongTensor).cuda()
                
                # eval mode changes behavior of dropout and batch norm for validation
                
                model.train(phase == 0)
                probs = model(data)

                label = torch.argmax(label, dim = 1)
                preds = torch.argmax(model.softmax(probs), dim = 1)

                loss = criterion(probs, label)

                optimizer.zero_grad()

                # backprop if in training phase
                if phase == 0:
                    loss.backward()
                    optimizer.step()

                    # i dont think it makes a huge difference but should i be doing these steps
                    # after both training and validation? since the weights get updated, then 
                    # the train accuracy/loss will be different than the val accuracy/loss

                    if scheduler is not None:
                        scheduler.step()

                running_loss += (loss.item() * len(data))
                running_correct += (preds == label).sum().item()

            # get metrics over entire dataset
            # need to make sure these calculations are correct
            true_accuracy = 100 * running_correct / len(dataset.get_subset(phase))
            true_loss = running_loss / len(dataset.get_subset(phase))

            if phase == 0:
                print("Epoch %d/%d, train accuracy: %.2f, train loss: %.4f" % (epoch, NUM_EPOCHS, true_accuracy, true_loss), end ="") 
            elif phase == 1:
                print(", val accuracy: %.2f, val loss: %.4f" % (true_accuracy, true_loss))

            # add metrics to list to be graphed
            if phase < 2:
                accuracy[phase].append(true_accuracy)
                losses[phase].append(true_loss)

        if epoch % GRAPH_FREQ == 0:
            grapher.update() 

        # output model weights to checkpoints directory if specified
        if epoch % SAVE_FREQ == 0:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }


            path = os.path.join('checkpoints', '%s_epoch_%d.t7' % (model.identifier, epoch))
            torch.save(state, path)

    grapher.show()

if __name__ == '__main__':
    main()
