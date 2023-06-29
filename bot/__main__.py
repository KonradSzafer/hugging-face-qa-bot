from bot.config import Config
from bot.logger import logger
from bot.discord_client import DiscordClient


def main():
    logger.info('Starting Application...')
    config = Config()
    client = DiscordClient(
        qa_service_url=config.qa_service_url,
        num_last_messages=config.num_last_messages,
        use_names_in_context=config.use_names_in_context,
        enable_commands=config.enable_commands,
        debug=config.debug
    )
    client.run(config.discord_token)

if __name__ == '__main__':
    main()
