import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define log file path
log_file = os.path.join(log_dir, "sentiment_analysis.log")

# Create a logger
logger = logging.getLogger("SentimentAnalysisLogger")
logger.setLevel(logging.DEBUG)

# Formatter for log messages
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Console handler (prints to stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler (writes logs to file, with rotation)
file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Logging at different levels
def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)

def log_debug(message):
    logger.debug(message)