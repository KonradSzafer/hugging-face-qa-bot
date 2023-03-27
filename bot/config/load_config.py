import os
from typing import Union
from dataclasses import dataclass


def get_env(env_name: str, default = None) -> str:
    env = os.getenv(env_name)
    if not env:
        if default:
            return default
        else:
            raise ValueError(f'Cannot parse: {env_name}')
    else:
        return env


@dataclass
class Config:
    huggingface_token: str = get_env('HUGGINGFACE_TOKEN')
    question_answering_model_id: str = get_env('QUESTION_ANSWERING_MODEL_ID')
    embedding_model_id: str = get_env('EMBEDDING_MODEL_ID')
    discord_token: str = get_env('DISCORD_TOKEN')
    num_last_messages: int = get_env('NUM_LAST_MESSAGES', 5)
    use_names_in_context: bool = get_env('USE_NAMES_IN_CONTEXT', True)
    enable_commands: bool = get_env('ENABLE_COMMANDS', True)
