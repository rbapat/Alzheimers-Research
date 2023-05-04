import torch.nn.functional as F
from datetime import datetime
from models import *
import torch.nn as nn
import numpy as np
import argparse
import dataset
import logger
import random
import torch
import os

class Constants:
    CLASSIFICATION = (1 << 0)
    REGRESSION = (1 << 1)
    SINGLE_TIMEPOINT = (1 << 2)
    LONGITUDINAL = (1 << 3)

    # OPERATION = CLASSIFICATION | SINGLE_TIMEPOINT
    OPERATION = CLASSIFICATION | LONGITUDINAL
    DX_CAP = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 1000
    IN_DIMS = (182, 218, 182)
    OUTPUT_DIM = 2

    DATASET_PATH = '/home/jupyter/Combined_FSL'
    EMBEDDING_PATH = '/home/jupyter/Embedding'
    SPLITS = [0.8, 0.2]
    LOAD_PATHS = False
    
    CLIN_VARS = ['MMSE', 'CDRSB', 'mPACCtrailsB', 'mPACCdigit', 'APOE4', 'ADAS11', 'ADAS13', 'ADASQ4', 'FAQ', 'RAVLT_forgetting', 'RAVLT_immediate', 'RAVLT_learning', 'TRABSCOR', 'Month']
    VISIT_DELTA = 6
    NUM_VISITS = 3

def main(c):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(c) 
    log = logger.Logger(c)

    model = DenseNet(c.IN_DIMS, c.OUTPUT_DIM, [6, 12, 32, 24], growth_rate = 24, theta = 0.5, drop_rate = 0.0).cuda()
    net = MultiModalNet # MultiModalNet, ImageOnly, CVOnly
    model = net(288, len(c.CLIN_VARS)).cuda()
    # model = DilationNet(c.IN_DIMS, c.OUTPUT_DIM).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)

    if c.OPERATION & c.CLASSIFICATION:
        criterion = nn.CrossEntropyLoss()
    elif c.OPERATION & c.REGRESSION:
        criterion = nn.MSELoss()
    else:
        raise RuntimeError(f"Unknown Operation: {c.OPERATION}")

    for epoch in range(1, c.NUM_EPOCHS + 1):
        log.epoch_start()

        for phase, loader in enumerate(parser.loaders):
            for mat, clin_vars, ground_truth in loader:
                mat, clin_vars, ground_truth = mat.cuda(), clin_vars.cuda(), ground_truth.view(ground_truth.shape[0]).cuda()
                #ground_truth = ground_truth.view(ground_truth.shape[0])

                optimizer.zero_grad()
                model.train(phase == 0)

                raw_output = model(mat, clin_vars)
                loss = criterion(raw_output, ground_truth)

                # track the number of correct classifications, the loss, and the number of patients that were predicted as 0 (the first class) - this is just used for debugging
                log.update(raw_output, ground_truth, loss, phase, len(mat), parser)
 
                if phase == 0:
                    loss.backward()
                    optimizer.step()

        log.epoch_end(c, epoch, model, optimizer)
    log.report()

if __name__ == '__main__':
    main(Constants())
