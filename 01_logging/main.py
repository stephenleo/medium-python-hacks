import constants as cst
import logging
import test_print

# Setup Basic Configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Instantiate the Logger
logger = logging.getLogger(cst.LOGGER_NAME)

# Create file handler and set level to custom level above CRITICAL
logging.addLevelName(cst.METRICS, 'METRICS')
file_handler = logging.FileHandler('metrics.log')
file_handler.setLevel(cst.METRICS)

# Add file handler to logger
logger.addHandler(file_handler)

if __name__ == '__main__':
    # Test out the logger from the original script
    logger.info(
        'This message is displayed on the terminal but NOT written to file')
    logger.log(
        cst.METRICS,
        'This message is both displayed on the terminal AND written to file')

    # Test out the logger in an imported function
    test_print.test_print()