from dotenv import load_dotenv
from api.logger import setup_logger


setup_logger()
load_dotenv(dotenv_path='config/api/.env')
