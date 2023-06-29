import os
from dataclasses import dataclass, asdict
from typing import Dict, Union
from api.logger import logger


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
    huggingface_token: str = get_env('HUGGINGFACEHUB_API_TOKEN')
    question_answering_model_id: str = get_env('QUESTION_ANSWERING_MODEL_ID')
    embedding_model_id: str = get_env('EMBEDDING_MODEL_ID')
    index_name: str = get_env('INDEX_NAME')
    use_docs_for_context: bool = eval(get_env('USE_DOCS_FOR_CONTEXT', 'True'))
    add_sources_to_response: bool = eval(get_env('ADD_SOURCES_TO_RESPONSE', 'True'))
    use_messages_in_context: bool = eval(get_env('USE_MESSAGES_IN_CONTEXT', 'True'))
    debug: bool = eval(get_env('DEBUG', 'True'))

    def __post_init__(self):
        # validate config
        if not self.use_docs_for_context and self.add_sources_to_response:
            raise ValueError('Cannot add sources to response if not using docs in context')
        self.log()

    def asdict(self) -> Dict:
        return asdict(self)

    def log(self) -> None:
        logger.info('Config:')
        for key, value in self.asdict().items():
            logger.info(f'{key}: {value}')
