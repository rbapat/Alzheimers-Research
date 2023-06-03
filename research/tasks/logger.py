import datetime
import logging
import sys
import os

import torch


class Logger:
    def __init__(
        self,
        log_folder: str,
        log_format: str,
        ckpt_folder: str,
        result_folder: str,
        log_epochs: bool,
    ):
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        cur_time = datetime.datetime.now()
        time_str = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_folder, f"{time_str}.txt")

        args = {
            "filename": log_filename,
            "format": "%(asctime)s %(message)s",
            "datefmt": "[%I:%M:%S] ",
            "level": logging.DEBUG,
        }

        logging.basicConfig(**args)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.info("Logger Initialized")

        self.log_epochs = log_epochs
        self.ckpt_filename = os.path.join(ckpt_folder, time_str, "{}.pt")
        self.result_filename = os.path.join(result_folder, time_str, "{}.pt")

        for filename in (self.ckpt_filename, self.result_filename):
            save_folder = os.path.dirname(filename)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

    def epoch(self, epoch_num, total_epochs, train_entry, val_entry, state_dict):
        if self.log_epochs:
            logging.info(
                f"Epoch [{epoch_num}/{total_epochs}] Train Accuracy: {train_entry[1]:.2f}, Val Accuracy: {val_entry[1]:.2f}, Train Loss: {train_entry[0]:.2f}, Val Loss: {val_entry[0]:.2f}"
            )

        if state_dict is not None:
            torch.save(state_dict, self.ckpt_filename.format(epoch_num))

    def save_results(self, results: torch.Tensor, save_name: str):
        torch.save(results, self.result_filename.format(save_name))
