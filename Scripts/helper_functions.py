import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS_DIR = os.path.join(BASE_DIR, os.getenv('LOGS_DIR'))

# Ensure Logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Define log file path with timestamp to keep logs separate per run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOGS_DIR, f"mlops_mobile_price_prediction_{timestamp}.log")

# Create a custom logger
logger = logging.getLogger("MobilePricePredictionLogger")
logger.setLevel(logging.INFO)

# Create handlers - file handler and console handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_info(message, context="General"):
    """
    Logs an info-level message with optional context.
    """
    logger.info(f"[{context}] {message}")

def log_error(message, context="General"):
    """
    Logs an error-level message with optional context.
    """
    logger.error(f"[{context}] {message}")

def log_warning(message, context="General"):
    """
    Logs a warning-level message with optional context.
    """
    logger.warning(f"[{context}] {message}")
