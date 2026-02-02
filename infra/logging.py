"""
Logging setup for fin_assist.
"""
import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'market_assistant.log')


def setup_logging():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger('fin_assist')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s [%(module)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    # file handler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)
    logger.propagate = False
    # expose module-level
    global log
    log = logger
    return logger


# ensure a default logger is available on import
try:
    log
except NameError:
    log = setup_logging()
