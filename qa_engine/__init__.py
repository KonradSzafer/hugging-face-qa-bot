from dotenv import load_dotenv
from qa_engine.logger import setup_logger

setup_logger()
load_dotenv(dotenv_path='config/.env')

from .logger import setup_logger, logger
from .config import Config
from .qa_engine import QAEngine
