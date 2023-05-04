import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
import numpy as np
import dataset
import random
import torch
import os

from models import DenseNet, DenseNetX

BATCH_SIZE = 1
NUM_EPOCHS = 1000
IN_DIMS = (182, 218, 182)
OUTPUT_DIM = 4

'''
Experimental task I was messing around with. I wanted to see how well I can predict the clinical variables themselves just from the 3D MRI scans.
It didn't work very well but it may be because of the way I preprocess the data or the network itself.
'''

def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(0, BATCH_SIZE)

    # model = DenseNetX(IN_DIMS, OUTPUT_DIM, [4, 4, 4, 4, 4], growth_rate = 24, theta = 0.5, drop_rate = 0.1).cuda()
    #model = DenseNet(IN_DIMS, OUTPUT_DIM, [6, 6, 6, 6], growth_rate = 64, theta = 0.5, drop_rate = 0.1).cuda()
    model = DenseNet(IN_DIMS, OUTPUT_DIM, [4, 4, 4, 4], growth_rate = 24, theta = 0.5, drop_rate = 0.1).cuda()
    #0.67: model = DenseNet(IN_DIMS, OUTPUT_DIM, [4, 4, 4, 4], growth_rate = 24, theta = 0.5, drop_rate = 0.1).cuda()
    model.load_weights('weights.t7')

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, nesterov = True)
    criterion = nn.MSELoss()


    for epoch in range(1, NUM_EPOCHS + 1):
        losses = [0, 0]

        for phase, loader in enumerate(parser.loaders):
            for mat, clin_vars, ground_truth in loader:
                mat, clin_vars, ground_truth = mat.cuda(), clin_vars.cuda(), ground_truth.view(ground_truth.shape[0]).cuda()

                optimizer.zero_grad()
                model.train(phase == 0)

                raw_output = model(mat, clin_vars)

                loss = criterion(raw_output, clin_vars)

                losses[phase] += (loss.item() * BATCH_SIZE) / parser.lengths[phase]

                if phase == 0:
                    loss.backward()
                    optimizer.step()

        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict}
        path = os.path.join('checkpoints', f'epoch_{epoch}.t7')
        torch.save(state, path)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{NUM_EPOCHS}: train loss: {round(losses[0], 4)}, test loss: {round(losses[1], 4)}")



if __name__ == '__main__':
    main()