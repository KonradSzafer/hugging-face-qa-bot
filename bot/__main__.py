from bot.config import Config
from bot.logger import logger
from bot.discord_client import DiscordClient
from bot.question_answering import LangChainModel


def main():
    logger.info('Starting Application...')
    config = Config()
    model = LangChainModel(
        question_answering_model_id=config.question_answering_model_id,
        embedding_model_id=config.embedding_model_id,
        hf_api_key=config.huggingface_token,
        run_localy=config.run_localy
    )
    client = DiscordClient(
        model=model,
        num_last_messages=config.num_last_messages,
        use_names_in_context=config.use_names_in_context,
        enable_commands=config.enable_commands
    )
    client.run(config.discord_token)

if __name__ == '__main__':
    main()
