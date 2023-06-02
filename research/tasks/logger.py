import datetime
import logging
import sys
import os


class Logger:
    def __init__(log_folder: str, log_format: str):
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        cur_time = datetime.datetime.now()
        time_str = cur_time.strftime("%Y-%m-%d_%H-%M-%S")

        args = {
            "filename": os.path.join(log_folder, f"{time_str}.txt"),
            "format": "%(asctime)s %(message)s",
            "datefmt": "[%I:%M:%S] ",
            "level": logging.DEBUG,
        }

        logging.basicConfig(**args)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.info("Logger Initialized")
