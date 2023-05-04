import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
from config import Settings
from models import *
import torch.nn as nn
import numpy as np
import argparse
import dataset
import logger
import random
import torch
import os

def plot_metric(total_data, name):
    total_data = total_data[:, 1:, :]
    mean_data = np.mean(total_data, axis = 0)
    max_data = np.amax(total_data, axis = 0)
    min_data = np.amin(total_data, axis = 0)
    
    x_coords = list(range(len(mean_data)))

    plt.title(name)
    plt.xlabel("Epoch")
    plt.ylabel(name)

    plt.plot(mean_data[:, 0], c = 'g', label = f'Train {name}')
    plt.fill_between(x_coords, mean_data[:, 0], max_data[:, 0], color = 'g', alpha = 0.2)
    plt.fill_between(x_coords, mean_data[:, 0], min_data[:, 0], color = 'g', alpha = 0.2)

    plt.plot(mean_data[:, 1], c = 'r', label = f'Validation {name}')
    plt.fill_between(x_coords, mean_data[:, 1], max_data[:, 1], color = 'r', alpha = 0.2)
    plt.fill_between(x_coords, mean_data[:, 1], min_data[:, 1], color = 'r', alpha = 0.2)

    plt.legend()
    plt.savefig(f"graphs/{name}.png")

    plt.figure()


def evaluate_model(c, net, log, train_set, val_set):
    model = net(288, len(c.CLIN_VARS)).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = c.LEARNING_RATE, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, c.NUM_EPOCHS + 1):
        log.epoch_start()
        for phase, loader in enumerate((train_set, val_set)):
            for mat, clin_vars, ground_truth in loader:
                optimizer.zero_grad()
                model.train(phase == 0)

                raw_output = model(mat, clin_vars)
                loss = criterion(raw_output, ground_truth)

                # track the number of correct classifications, the loss, and the number of patients that were predicted as 0 (the first class) - this is just used for debugging
                log.update(raw_output, ground_truth, loss, phase, len(mat))

                if phase == 0:
                    loss.backward()
                    optimizer.step()

        log.epoch_end(c, epoch, model, optimizer)


    log.report()

def nested_cv_main(c):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(c) 
    log = logger.Logger()

    net = MultiModalNet # MultiModalNet, ImageOnly, CVOnly
    num_total_folds = parser.outer_skf.get_n_splits() * parser.inner_skf.get_n_splits()

    if c.CROSS_VAL:
        train_best_acc_list = ([], [], [])
        test_best_acc_list = ([], [], [])

        train_total_acc, train_total_spec, train_total_sens = np.zeros((num_total_folds, c.NUM_EPOCHS, 2)), np.zeros((num_total_folds, c.NUM_EPOCHS, 2)), np.zeros((num_total_folds, c.NUM_EPOCHS, 2))
        test_total_acc, test_total_spec, test_total_sens = np.zeros((parser.outer_skf.get_n_splits() , c.NUM_EPOCHS, 2)), np.zeros((parser.outer_skf.get_n_splits() , c.NUM_EPOCHS, 2)), np.zeros((parser.outer_skf.get_n_splits() , c.NUM_EPOCHS, 2))

        for outer_fold_idx, (outer_train_split, test_set) in enumerate(parser.cross_validation_set()):
            for inner_fold_idx, (train_set, val_set) in enumerate(parser.cross_validation_set(outer_train_split)):
                print(f"Starting outer fold {outer_fold_idx}, inner fold {inner_fold_idx}")   

                log.reset_tracking()
                evaluate_model(c, net, log, train_set, val_set)

                train_best_acc_list[0].append(log.best_acc[1])
                train_best_acc_list[1].append(log.best_sens[1])
                train_best_acc_list[2].append(log.best_spec[1])

                train_total_acc[outer_fold_idx * parser.inner_skf.get_n_splits() + inner_fold_idx, :] = np.array(log.total_accuracy)
                train_total_spec[outer_fold_idx * parser.inner_skf.get_n_splits() + inner_fold_idx, :] = np.array(log.total_specificity)
                train_total_sens[outer_fold_idx * parser.inner_skf.get_n_splits() + inner_fold_idx, :] = np.array(log.total_sensitivity)

            log.reset_tracking()
            full_train = parser.full_training_set(outer_train_split)
            evaluate_model(c, net, log, full_train, test_set) 

            test_best_acc_list[0].append(log.best_acc[1])
            test_best_acc_list[1].append(log.best_sens[1])
            test_best_acc_list[2].append(log.best_spec[1])

            test_total_acc[outer_fold_idx, :] = np.array(log.total_accuracy)
            test_total_spec[outer_fold_idx, :] = np.array(log.total_specificity)
            test_total_sens[outer_fold_idx, :] = np.array(log.total_sensitivity)



        plot_metric(train_total_acc, "Accuracy")
        plot_metric(train_total_spec, "Specificity")
        plot_metric(train_total_sens, "Sensitivity")

        print("----------------------------- Average CV Metrics-----------------------------")
        print(f"Fold Accuracies: {train_best_acc_list[0]}, average: {round(np.mean(train_best_acc_list[0]), 4)}, std: {round(np.std(train_best_acc_list[0]), 4)}")
        print(f"Fold Sensitivities: {train_best_acc_list[1]}, average: {round(np.mean(train_best_acc_list[1]), 4)}, std: {round(np.std(train_best_acc_list[1]), 4)}")
        print(f"Fold Specificities: {train_best_acc_list[2]}, average: {round(np.mean(train_best_acc_list[2]), 4)}, std: {round(np.std(train_best_acc_list[2]), 4)}")
        print("----------------------------- Average Test Metrics --------------------------")        
        print(f"Fold Accuracies: {test_best_acc_list[0]}, average: {round(np.mean(test_best_acc_list[0]), 4)}, std: {round(np.std(test_best_acc_list[0]), 4)}")
        print(f"Fold Sensitivities: {test_best_acc_list[1]}, average: {round(np.mean(test_best_acc_list[1]), 4)}, std: {round(np.std(test_best_acc_list[1]), 4)}")
        print(f"Fold Specificities: {test_best_acc_list[2]}, average: {round(np.mean(test_best_acc_list[2]), 4)}, std: {round(np.std(test_best_acc_list[2]), 4)}")
        print("-----------------------------------------------------------------------------")
        

    else:
        train = parser.full_training_set()
        val = parser.full_testing_set()

        evaluate_model(c, net, log, train, val)
        print(f"Best accuracy reported as {log.best_acc[1]}")

def flat_cv_main(c):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(c) 
    log = logger.Logger()

    net = CVOnly # MultiModalNet, ImageOnly, CVOnly

    if c.CROSS_VAL:
        best_acc_list = ([], [], [])
        total_acc, total_spec, total_sens = np.zeros((3, c.NUM_EPOCHS, 2)), np.zeros((3, c.NUM_EPOCHS, 2)), np.zeros((3, c.NUM_EPOCHS, 2))

        for fold_idx, (train_set, val_set) in enumerate(parser.cross_validation_set()):
            print(f"Starting fold {fold_idx}...")

            log.reset_tracking()
            evaluate_model(c, net, log, train_set, val_set)

            best_acc_list[0].append(log.best_acc[1])
            best_acc_list[1].append(log.best_sens[1])
            best_acc_list[2].append(log.best_spec[1])

            total_acc[fold_idx, :] = np.array(log.total_accuracy)
            total_spec[fold_idx, :] = np.array(log.total_specificity)
            total_sens[fold_idx, :] = np.array(log.total_sensitivity)

        plot_metric(total_acc, "Accuracy")
        plot_metric(total_spec, "Specificity")
        plot_metric(total_sens, "Sensitivity")

        print(f"Fold Accuracies: {best_acc_list[0]}, average: {round(np.mean(best_acc_list[0]), 4)}, std: {round(np.std(best_acc_list[0]), 4)}")
        print(f"Fold Sensitivities: {best_acc_list[1]}, average: {round(np.mean(best_acc_list[1]), 4)}, std: {round(np.std(best_acc_list[1]), 4)}")
        print(f"Fold Specificities: {best_acc_list[2]}, average: {round(np.mean(best_acc_list[2]), 4)}, std: {round(np.std(best_acc_list[2]), 4)}")

    else:
        train = parser.full_training_set()
        val = parser.full_testing_set()

        evaluate_model(c, net, log, train, val)
        print(f"Best accuracy reported as {log.best_acc[1]}")

if __name__ == '__main__':
    if Settings.OUTER_SPLIT == 1:
        flat_cv_main(Settings)
    else:
        nested_cv_main(Settings)
