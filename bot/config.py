import os
from dataclasses import dataclass, asdict
from typing import Dict, Union
from bot.logger import logger


def get_env(env_name: str, default = None) -> str:
    env = os.getenv(env_name)
    if not env:
        if default:
            logger.warning(
                f'Environment variable {env_name} not found.' \
                f'Using the default value: {default}.'
            )
            return default
        else:
            raise ValueError(f'Cannot parse: {env_name}')
    else:
        return env


@dataclass
class Config:
    discord_token: str = get_env('DISCORD_TOKEN')
    qa_service_url: str = get_env('QA_SERVICE_URL')
    
    use_messages_in_context: bool = eval(get_env('USE_MESSEGES_IN_CONTEXT', 'True'))
    num_last_messages: int = int(get_env('NUM_LAST_MESSAGES', 2))
    use_names_in_context: bool = eval(get_env('USE_NAMES_IN_CONTEXT', 'False'))
    enable_commands: bool = eval(get_env('ENABLE_COMMANDS', 'True'))
    debug: bool = eval(get_env('DEBUG', 'True'))

    def __post_init__(self):
        # validate config
        self.log()

    def asdict(self) -> Dict:
        return asdict(self)

    def log(self) -> None:
        logger.info('Config:')
        for key, value in self.asdict().items():
            logger.info(f'{key}: {value}')
