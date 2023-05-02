import os
from dataclasses import dataclass, asdict
from typing import Dict, Union
from bot.logger import logger


def get_env(env_name: str, default = None) -> str:
    """
    Retrieve an environment variable.

    Args:
        env_name (str): The name of the environment variable.
        default (optional): The default value to return if the environment variable is not found. Defaults to None.

    Returns:
        str: The value of the environment variable.
    """
    env = os.getenv(env_name)
    if not env:
        if default:
            return default
        else:
            raise ValueError(f'Cannot parse: {env_name}')
    else:
        return env

def to_bool(variable: str) -> bool:
    """
    Convert a string to a boolean.

    Args:
        variable (str): The string to be converted.

    Returns:
        bool: The boolean value of the string.
    """
    if variable.lower() in ['true', '1']:
        return True
    elif variable.lower() in ['false', '0']:
        return False
    else:
        raise ValueError(f'Cannot parse: {variable}')

@dataclass
class Config:
    """
    Configuration class, used for setting up the application.

    Attributes:
        huggingface_token (str): The Hugging Face API token.
        discord_token (str): The Discord API token.
        question_answering_model_id (str): The ID of the question answering model to be used.
        embedding_model_id (str): The ID of the embedding model to be used.
        index_name (str): The name of the FAISS index to be used.
        use_docs_in_context (bool): Whether to use relevant documents as context for generating answers.
        use_messages_in_context (bool): Whether to use previous messages as context for generating answers.
        num_last_messages (int): The number of previous messages to use as context for generating answers.
        use_names_in_context (bool): Whether to include user names in the message context.
        enable_commands (bool): Whether to enable commands for the bot.
        run_locally (bool): Whether to run the application locally or on the Hugging Face hub.

    Methods:
        asdict(self) -> Dict: Return the configuration as a dictionary.
        log(self) -> None: Log the configuration.
    """
    huggingface_token: str = get_env('HUGGINGFACEHUB_API_TOKEN')
    discord_token: str = get_env('DISCORD_TOKEN')
    question_answering_model_id: str = get_env('QUESTION_ANSWERING_MODEL_ID')
    embedding_model_id: str = get_env('EMBEDDING_MODEL_ID')
    index_name: str = get_env('INDEX_NAME')
    use_docs_in_context: bool = to_bool(get_env('USE_DOCS_IN_CONTEXT', True))
    use_messages_in_context: bool = to_bool(get_env('USE_MESSEGES_IN_CONTEXT', True))
    num_last_messages: int = int(get_env('NUM_LAST_MESSAGES', 2))
    use_names_in_context: bool = to_bool(get_env('USE_NAMES_IN_CONTEXT', False))
    enable_commands: bool = to_bool(get_env('ENABLE_COMMANDS', True))
    run_locally: bool = to_bool(get_env('RUN_LOCALLY', True))

    def asdict(self) -> Dict:
        return asdict(self)

    def log(self) -> None:
        logger.info('Config:')
        for key, value in self.asdict().items():
            logger.info(f'{key}: {value}')
