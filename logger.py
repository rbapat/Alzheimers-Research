import os
import torch
import sys
from datetime import datetime
import torch.nn.functional as F
import logging
from sklearn import metrics

class Logger:
    def __init__(self):
        os.makedirs('checkpoints', exist_ok = True)

        self.init_logging()
        self.reset_tracking()

    def init_logging(self, log_path = 'network_log.txt'):
        if os.path.exists(log_path):
            os.remove(log_path)

        args = {
            'filename': log_path,
            'format': '%(asctime)s %(message)s',
            'datefmt': '[%I:%M:%S] ',
            'level': logging.DEBUG
        }

        logging.basicConfig(**args)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.info("Logger Initialized")

    def reset_tracking(self):
        self.losses = [0, 0]
        self.num_in_current = [0, 0]

        self.running_preds = [[], []]
        self.running_corrects = [[], []]

        self.total_accuracy = []
        self.total_sensitivity = []
        self.total_specificity = []

        self.best_acc = [-1, -1]
        self.best_spec = [-1, -1]
        self.best_sens = [-1, -1]

    def update(self, raw_output, ground_truth, loss, phase, bs):

        self.losses[phase] += (loss.item() * bs)
        self.num_in_current[phase] += bs

        preds = torch.argmax(F.softmax(raw_output, dim = 1), dim = 1)

        for gt, p in zip(ground_truth, preds):
            self.running_corrects[phase].append(gt.item())
            self.running_preds[phase].append(p.item())

    def epoch_start(self):
        self.losses = [0, 0]
        self.num_in_current = [0, 0]

        self.running_preds = [[], []]
        self.running_corrects = [[], []]

    def get_metrics(self):
        bal_acc, sens, spec = [], [], []
        
        for correct, pred in zip(self.running_corrects, self.running_preds):
            bal_acc.append(round(metrics.balanced_accuracy_score(correct, pred), 4))
            sens.append(round(metrics.recall_score(correct, pred, pos_label = 1), 4))
            spec.append(round(metrics.recall_score(correct, pred, pos_label = 0), 4))

        return bal_acc, sens, spec

    def epoch_end(self, c, epoch, model, optimizer):
        # state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict}
        # path = os.path.join('checkpoints', f'epoch_{epoch}.t7')
        # torch.save(state, path)

        losses = [round(l / nic, 4) for l, nic in zip(self.losses, self.num_in_current)]
        bal_acc, sens, spec = self.get_metrics()

        self.total_accuracy.append(bal_acc)
        self.total_sensitivity.append(sens)
        self.total_specificity.append(spec)

        if bal_acc[1] > self.best_acc[1]:
            self.best_acc = [epoch, bal_acc[1]]
            self.best_sens = [epoch, sens[1]]
            self.best_spec = [epoch, spec[1]]

        msg = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{c.NUM_EPOCHS}: train loss: {losses[0]}, train accuracy: {bal_acc[0]}%, test loss: {losses[1]}, test accuracy: {bal_acc[1]}% | sn[{sens[1]}] sp[{spec[1]}]"
        logging.info(msg)

    def report(self):
        logging.info(f"Best Accuracy was at epoch {self.best_acc[0]} with a value of {self.best_acc[1]}")
        logging.info(f"Best Sensitivity was at epoch {self.best_sens[0]} with a value of {self.best_sens[1]}")
        logging.info(f"best Specificity was at epoch {self.best_spec[0]} with a value of {self.best_spec[1]}")
