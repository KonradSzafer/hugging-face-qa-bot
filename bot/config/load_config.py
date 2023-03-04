import os
from dataclasses import dataclass


def get_env(env_name: str) -> str:
    env = os.getenv(env_name)
    if not env:
        raise ValueError(f'Cannot parse: {env_name}')
    else:
        return env


@dataclass
class Config:
    discord_token: str = get_env('DISCORD_TOKEN')
    discord_guild: str = get_env('DISCORD_GUILD')
