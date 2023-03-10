from bot.config import Config
from bot.logger import logger
from bot.discord_client import DiscordClient
from bot.question_answering import LangChainModel


def main():
    logger.info('Starting Application...')
    config = Config()
    model = LangChainModel(
        config.huggingface_token,
        config.huggingface_model_id
    )
    client = DiscordClient(model)
    client.run(config.discord_token)

if __name__ == '__main__':
    main()
