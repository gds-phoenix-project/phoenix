import logging
import os
import time
from logging.handlers import RotatingFileHandler

import colorlog

LOGGER_CONFIG_PATH = "logs"

# Provides global log output and retention
def init_logger():
    log_level = logging.INFO
    log_colors_config = {
        'DEBUG': 'bold_cyan',
        'INFO': 'bold_green',
        'WARNING': 'bold_yellow',
        'ERROR': 'bold_red',
        'CRITICAL': 'red',
    }
    fmts = {
        "console": "%(log_color)s[%(levelname)s] %(log_color)s%(asctime)s [%(filename)s:%(lineno)d] : %(message)s",
        "file": "[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] : %(message)s"
    }
    color_formatter = colorlog.ColoredFormatter(
        fmt=fmts["console"], reset=True, log_colors=log_colors_config,
        secondary_log_colors={
            'message': {
                'DEBUG': 'blue',
                'INFO': 'blue',
                'WARNING': 'blue',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
            }
        }, style='%'
    )
    
    root_path = os.path.dirname(os.path.realpath(__file__))
    LOGGER_CONFIG_PATH = os.path.join(root_path, "logs")
    
    if not os.path.exists(LOGGER_CONFIG_PATH):
        os.mkdir(LOGGER_CONFIG_PATH)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOGGER_CONFIG_PATH, "{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))),
        maxBytes=1024 * 1024 * 50, backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(fmts["file"]))

    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(color_formatter)

    logger = logging.getLogger("nds")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


Log = init_logger()
