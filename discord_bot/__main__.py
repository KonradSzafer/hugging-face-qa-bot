from qa_engine import logger, Config, QAEngine
from discord_bot.client import DiscordClient


config = Config()
qa_engine = QAEngine(config=config)
client = DiscordClient(
    qa_engine=qa_engine,
    config=config
)


if __name__ == '__main__':
    logger.info('Starting Application...')
    client.run(config.discord_token)
