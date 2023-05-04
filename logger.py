import os
import torch
import sys
from datetime import datetime
import torch.nn.functional as F
import logging
from sklearn import metrics

class Logger:
    def __init__(self, c):
        self.c = c

        os.makedirs('checkpoints', exist_ok = True)
        self.init_logging()

        self.losses = [0, 0]

        self.preds = [[], []]
        self.corrects = [[], []]

        self.best_acc = [-1, -1]
        self.best_spec = [-1, -1]
        self.best_sens = [-1, -1]

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
        logging.info("Logger Initialized")

    def update(self, raw_output, ground_truth, loss, phase, bs, parser):
        self.losses[phase] += (loss.item() * bs) / parser.lengths[phase]

        if self.c.OPERATION & self.c.CLASSIFICATION:
            preds = torch.argmax(F.softmax(raw_output, dim = 1), dim = 1)

            for gt, p in zip(ground_truth, preds):
                self.corrects[phase].append(gt.item())
                self.preds[phase].append(p.item())

            # self.corrects[phase] += 100 * (preds == ground_truth).sum().item() / parser.lengths[phase]

    def epoch_start(self):
        self.losses = [0, 0]

        self.preds = [[], []]
        self.corrects = [[], []]

    def get_metrics(self):
        ba, rc, pr, sn, sp = [], [], [], [], []
        
        for correct, pred in zip(self.corrects, self.preds):
            ba.append(round(metrics.balanced_accuracy_score(correct, pred), 4))
            report = metrics.classification_report(correct, pred, target_names=['ncvt', 'cvt'], output_dict = True)
            
            rc.append(round(report['cvt']['recall'], 2))
            pr.append(round(report['cvt']['precision'], 2))
            sn.append(round(report['cvt']['recall'], 2))
            sp.append(round(report['ncvt']['recall'], 2))

            # rc.append(round(metrics.recall_score(correct, pred), 4))
            # pr.append(round(metrics.precision_score(correct, pred), 4))
            # sn.append(round(metrics.recall_score(correct, pred, pos_label = 0), 4))
            # sp.append(round(metrics.recall_score(correct, pred, pos_label = 0), 4))

        return ba, sn, sp

    def epoch_end(self, c, epoch, model, optimizer):
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict}
        path = os.path.join('checkpoints', f'epoch_{epoch}.t7')
        torch.save(state, path)

        losses = [round(l, 4) for l in self.losses]
        bal_acc, sens, spec = self.get_metrics()

        if bal_acc[1] > self.best_acc[1]:
            self.best_acc = [epoch, bal_acc[1]]
        if sens[1] > self.best_sens[1]:
            self.best_sens = [epoch, sens[1]]
        if spec[1] > self.best_spec[1]:
            self.best_spec = [epoch, spec[1]]

        if self.c.OPERATION & self.c.CLASSIFICATION:
            msg = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{c.NUM_EPOCHS}: train loss: {losses[0]}, train accuracy: {bal_acc[0]}%, test loss: {losses[1]}, test accuracy: {bal_acc[1]}% | ba[{bal_acc[1]}] sn[{sens[1]}] sp[{spec[1]}]"
        else:
            msg = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{c.NUM_EPOCHS}: train loss: {losses[1]}, test loss: {losses[1]}"

        logging.info(msg)

    def report(self):
        logging.info(f"Best Accuracy was at epoch {self.best_acc[0]} with a value of {self.best_acc[1]}")
        logging.info(f"Best Sensitivity was at epoch {self.best_sens[0]} with a value of {self.best_sens[1]}")
        logging.info(f"best Specificity was at epoch {self.best_spec[0]} with a value of {self.best_spec[1]}")
