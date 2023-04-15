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

def to_bool(variable: str) -> bool:
    if variable.lower() in ['true', '1']:
        return True
    elif variable.lower() in ['false', '0']:
        return False
    else:
        raise ValueError(f'Cannot parse: {variable}')

@dataclass
class Config:
    huggingface_token: str = get_env('HUGGINGFACEHUB_API_TOKEN')
    question_answering_model_id: str = get_env('QUESTION_ANSWERING_MODEL_ID')
    embedding_model_id: str = get_env('EMBEDDING_MODEL_ID')
    discord_token: str = get_env('DISCORD_TOKEN')
    num_last_messages: int = int(get_env('NUM_LAST_MESSAGES', 10))
    use_names_in_context: bool = to_bool(get_env('USE_NAMES_IN_CONTEXT', False))
    enable_commands: bool = to_bool(get_env('ENABLE_COMMANDS', True))
    run_locally: bool = to_bool(get_env('RUN_LOCALLY', True))
