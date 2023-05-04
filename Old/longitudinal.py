from sklearn.model_selection import ParameterGrid
import torch.nn.functional as F
from datetime import datetime
from pprint import pprint
import torch.nn as nn
import numpy as np
import dataset
import random
import torch
import os

from models import DenseNet, LSTMNet

NUM_EPOCHS = 800

# trains model on the longitudinal classification task: predicting conversion from MCI to AD
def evaluate_model(params):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(1, params['batch_size'])

    model = model = LSTMNet(params['conv_features'], params['lstm_hidden'], params['lstm_layers'], params['make_predictor'][0], params['drop_rate']).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = params['learning_rate'], momentum = 0.9, nesterov = True, weight_decay = 0.001)
    criterion = nn.CrossEntropyLoss()

    good_epochs = []
    model_history = ([], [], [], [])
    for epoch in range(1, NUM_EPOCHS + 1):
        nz, corrects, losses = 0, [0, 0], [0, 0]

        for phase, loader in enumerate(parser.loaders):
            for mat, clin_vars, ground_truth in loader:
                ground_truth = ground_truth.view(ground_truth.shape[0]).cuda()
                optimizer.zero_grad()
                model.train(phase == 0)

                # even though I pass in clin_vars, I'm not really using clinical variables right now; just something I was experimenting with
                raw_output = model(mat, clin_vars)
                
                loss = criterion(raw_output, ground_truth)

                # track the number of correct classifications, the loss, and the number of patients that were predicted as 0 (the first class) - this is just used for debugging
                preds = torch.argmax(F.softmax(raw_output, dim = 1), dim = 1)
                losses[phase] += (loss.item() * len(mat)) / parser.lengths[phase]
                corrects[phase] += 100 * (preds == ground_truth).sum().item() / parser.lengths[phase]
                nz += torch.count_nonzero(preds)

                if phase == 0:
                    loss.backward()
                    optimizer.step()

        # Save model weights
        
        '''
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict}
        path = os.path.join('checkpoints', f'epoch_{epoch}.t7')
        torch.save(state, path)
        '''
        
        if corrects[1] > 76:
            state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict}
            path = os.path.join('checkpoints', f'epoch-{epoch}_acc-{round(corrects[1], 3)}_loss-{round(losses[1], 3)}.t7')
            torch.save(state, path)
            good_epochs.append((epoch, losses[1], corrects[1]))

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{NUM_EPOCHS}: train loss: {round(losses[0], 4)}, train accuracy: {round(corrects[0], 4)}%, test loss: {round(losses[1], 4)}, test accuracy: {round(corrects[1], 4)}% | {nz}")
        model_history[0].append(losses[0])
        model_history[1].append(losses[1])
        model_history[2].append(corrects[0])
        model_history[3].append(corrects[1])

    pprint(params)
    pprint(good_epochs)
    print(max(model_history[3]))
    return model_history


def grid_search():

    m0 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, out_dims))
    m1 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, out_dims))
    m2 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(), nn.Linear(lstm_hidden // 4, out_dims))
    m3 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(), nn.Linear(lstm_hidden // 4, lstm_hidden // 8), nn.ReLU(), nn.Linear(lstm_hidden // 8, out_dims))    

    param_grid = list(ParameterGrid({
        'lstm_hidden': [512, 256, 64],
        'lstm_layers': [1,2,3],
        'conv_features': [1000],
        'batch_size': [64],
        'learning_rate': [0.01],
        'make_predictor': [(m0, 'm0'), (m1, 'm1'), (m2, 'm2'), (m3, 'm3')],
        'drop_rate': [0.0, 0.1, 0.2, 0.3]
    }))
        
    for params in param_grid:
        evaluate_model(params)
        print()

def main():
    m0 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, out_dims))
    m1 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, out_dims))
    m2 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(), nn.Linear(lstm_hidden // 4, out_dims))
    m3 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(), nn.Linear(lstm_hidden // 4, lstm_hidden // 8), nn.ReLU(), nn.Linear(lstm_hidden // 8, out_dims))

    params = {
        'lstm_hidden': 1024,
        'lstm_layers': 2,
        'conv_features': 1000,
        'batch_size': 64,
        'learning_rate': 0.01,
        'make_predictor': (m2,'m2'),
        'drop_rate': 0.1
    }

    evaluate_model(params)

if __name__ == '__main__':
    main()