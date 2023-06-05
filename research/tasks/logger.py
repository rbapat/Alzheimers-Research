import datetime
import logging
import sys
import os

import torch


class Logger:
    def __init__(
        self,
        ckpt_folder: str,
        result_folder: str,
        log_epochs: bool,
        **kwargs,
    ):
        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)

        cur_time = datetime.datetime.now()
        time_str = cur_time.strftime("%Y-%m-%d_%H-%M-%S")

        args = {
            "format": "%(asctime)s %(message)s",
            "datefmt": "[%I:%M:%S] ",
            "level": logging.DEBUG,
        }

        logging.basicConfig(**args)
        # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.info(f"Logger initialized")

        self.log_epochs = log_epochs
        self.ckpt_filename = os.path.join(ckpt_folder, time_str, "{}.pt")
        self.result_filename = os.path.join(result_folder, time_str, "{}.pt")

        for filename in (self.ckpt_filename, self.result_filename):
            save_folder = os.path.dirname(filename)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)

    def save_weights(self, model: torch.nn.Module, path: str):
        torch.save(model.state_dict(), path)

    def load_weights(self, model: torch.nn.Module, path: str):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt)

    def epoch(self, epoch_num, total_epochs, train_entry, val_entry, model):
        if self.log_epochs:
            logging.info(
                f"Epoch [{epoch_num}/{total_epochs}] Train Accuracy: {train_entry[1]*100:.2f}%, Val Accuracy: {val_entry[1]*100:.2f}%, Train Loss: {train_entry[0]:.4f}, Val Loss: {val_entry[0]:.4f}"
            )

        if model is not None:
            self.save(model, self.ckpt_filename.format(epoch_num))

    def save_results(self, results: torch.Tensor, save_name: str):
        torch.save(results, self.result_filename.format(save_name))
