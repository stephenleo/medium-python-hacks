import constants as cst
import logging

# Instantiate the same logger
logger = logging.getLogger(cst.LOGGER_NAME)

def test_print():
    logger.log(cst.METRICS, 'This also writes to file!')