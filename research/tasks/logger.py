from typing import List, Tuple
import datetime
import logging
import os


import torch
import matplotlib.pyplot as plt


class Logger:
    def __init__(
        self,
        ckpt_folder: str,
        result_folder: str,
        figure_folder: str,
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

        self.heat_vol_path = os.path.join(
            figure_folder, "heatmaps", "volumes", f"{time_str}.pt"
        )
        self.heat_image_path = os.path.join(
            figure_folder, "heatmaps", "images", time_str, "{}.svg"
        )

        logging_dirs = (
            self.ckpt_filename,
            self.result_filename,
            self.heat_vol_path,
            self.heat_image_path,
        )
        for filename in logging_dirs:
            save_folder = os.path.dirname(filename)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)

    def save_weights(self, model: torch.nn.Module, path: str):
        torch.save(model.state_dict(), path)

    def load_weights(self, model: torch.nn.Module, path: str):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt)

    def epoch_new(self, epoch_num, total_epochs, entries, names, model, extra_name=""):
        if self.log_epochs:
            msg = f"Epoch [{epoch_num}/{total_epochs}] "

            for entry, name in zip(entries, names):
                msg += f"{name} Accuracy: {entry[1]*100:.2f}%, "

            for entry, name in zip(entries, names):
                msg += f"{name} Loss: {entry[0]:.4f}, "

            logging.info(msg[:-2])

        if model is not None:
            self.save_weights(
                model, self.ckpt_filename.format(f"{extra_name}_epoch{epoch_num}")
            )

    def save_results(self, results: torch.Tensor, save_name: str):
        torch.save(results, self.result_filename.format(save_name))

    def save_heatvol(
        self, heatmaps: torch.Tensor, volumes: torch.Tensor, ptids: List[str]
    ) -> str:
        state_dict = {"heatmaps": heatmaps, "volumes": volumes, "ptids": ptids}

        torch.save(state_dict, self.heat_vol_path)
        return self.heat_vol_path

    def save_heatimage(self, figure_name) -> str:
        path = self.heat_image_path.format(figure_name)
        plt.savefig(path, bbox_inches="tight", pad_inches=0)

        return path

    def load_heatvol(self, path: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        heatvols = torch.load(path)
        return heatvols["heatmaps"], heatvols["volumes"], heatvols["ptids"]
