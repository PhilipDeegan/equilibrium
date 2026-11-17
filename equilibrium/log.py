import logging
import os

LOG_LEVEL = os.environ.get("EQUILIBRIUM_LOG_LEVEL", "INFO")
log_levels = {"INFO": 20, "WARNING": 30, "ERROR": 40, "DEBUG": 10}

FORMAT = "[log: %(filename)s:%(lineno)s - %(funcName)2s() ] %(message)s"
logging.basicConfig(format=FORMAT)

# logging.basicConfig(
#     format="%(asctime)s %(levelname)-8s %(message)s",
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )


def getLogger(name, level=LOG_LEVEL):
    logger = logging.getLogger(__name__)
    level = log_levels[level] if isinstance(level, str) else level
    logger.setLevel(level)
    return logger
