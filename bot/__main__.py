from bot.config import Config
from bot.logger import logger
from bot.discord_client import DiscordClient


config = Config()

def main():
    logger.info('Starting Application...')
    client = DiscordClient()
    client.run(config.discord_token)

if __name__ == '__main__':
    main()
