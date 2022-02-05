import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
import numpy as np
import dataset
import random
import torch
import os

from models import DenseNet, LSTMNet

BATCH_SIZE = 64
NUM_EPOCHS = 2000

# trains model on the longitudinal classification task: predicting conversion from MCI to AD
def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(1, BATCH_SIZE)

    model = model = LSTMNet().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, nesterov = True)
    criterion = nn.CrossEntropyLoss()

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

        '''
        # Save model weights
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict}
        path = os.path.join('checkpoints', f'epoch_{epoch}.t7')
        torch.save(state, path)
        '''

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{NUM_EPOCHS}: train loss: {round(losses[0], 4)}, train accuracy: {round(corrects[0], 4)}%, test loss: {round(losses[1], 4)}, test accuracy: {round(corrects[1], 4)}% | {nz}")



if __name__ == '__main__':
    main()