import logging


logger = logging.getLogger(__name__)

def setup_logger() -> None:
    """
    Logger setup.
    """
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
