import os
from dataclasses import dataclass, asdict
from typing import Any, Union

from qa_engine import logger


def get_env(env_name: str, default: Any = None, warn: bool = True) -> str:
    env = os.getenv(env_name)
    if not env:
        if default:
            if warn:
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
    # QA Engine config
    question_answering_model_id: str = get_env('QUESTION_ANSWERING_MODEL_ID')
    embedding_model_id: str = get_env('EMBEDDING_MODEL_ID')
    index_repo_id: str = get_env('INDEX_REPO_ID')
    prompt_template_name: str = get_env('PROMPT_TEMPLATE_NAME')
    use_docs_for_context: bool = eval(get_env('USE_DOCS_FOR_CONTEXT', 'True'))
    num_relevant_docs: bool = eval(get_env('NUM_RELEVANT_DOCS', 3))
    add_sources_to_response: bool = eval(get_env('ADD_SOURCES_TO_RESPONSE', 'True'))
    use_messages_in_context: bool = eval(get_env('USE_MESSAGES_IN_CONTEXT', 'True'))
    debug: bool = eval(get_env('DEBUG', 'True'))

    # Discord bot config - optional
    discord_token: str = get_env('DISCORD_TOKEN', '-', warn=False)
    num_last_messages: int = int(get_env('NUM_LAST_MESSAGES', 2, warn=False))
    use_names_in_context: bool = eval(get_env('USE_NAMES_IN_CONTEXT', 'False', warn=False))
    enable_commands: bool = eval(get_env('ENABLE_COMMANDS', 'True', warn=False))

    # App mode
    app_mode: str = get_env('APP_MODE', '-', warn=False) # 'gradio' or 'discord'

    def __post_init__(self):
        prompt_template_file = f'config/prompt_templates/{self.prompt_template_name}.txt'
        with open(prompt_template_file, 'r') as f:
            self.prompt_template = f.read()
        # validate config
        if 'context' not in self.prompt_template:
            raise ValueError("Prompt Template does not contain the 'context' field.")
        if 'question' not in self.prompt_template:
            raise ValueError("Prompt Template does not contain the 'question' field.")
        if not self.use_docs_for_context and self.add_sources_to_response:
            raise ValueError('Cannot add sources to response if not using docs in context')
        if self.num_relevant_docs < 1:
            raise ValueError('num_relevant_docs must be greater than 0')
        self.log()

    def asdict(self) -> dict:
        return asdict(self)

    def log(self) -> None:
        logger.info('Config:')
        for key, value in self.asdict().items():
            logger.info(f'{key}: {value}')
