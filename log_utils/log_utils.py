from datetime import datetime
import logging
from sys import stdout
import os

if not os.path.exists("log_files/"):
    os.mkdir("log_files/")

class CustomLogger:
    def __init__(self, log_filename: str):
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO, stream=stdout)
        logger.addHandler(
            logging.FileHandler(
                "log_files/log_{0}_{1}.txt".format(log_filename, datetime.now())
            )
        )
        self.logger = logger

    def write_log(self, message: str):
        timestamp = datetime.now()
        self.logger.info("[{0}]: {1}".format(timestamp, message))
