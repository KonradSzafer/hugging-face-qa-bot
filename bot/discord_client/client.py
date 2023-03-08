import discord
from bot.logger import logger


class DiscordClient(discord.Client):
    def __init__(self):
        logger.info('Initializing Discord client...')
        intents = discord.Intents.all()
        intents.message_content = True
        super().__init__(intents=intents, command_prefix='!')

    async def on_ready(self):
        logger.info('Successfully logged in as: {0.user}'.format(self))

    async def on_message(self, message):
        if message.author == self.user:
            return
        logger.info('Received message: {0.content}'.format(message))
        response = 'hello'
        logger.info('Sending response: {0}'.format(response))
        try:
            await message.channel.send(response)
        except Exception as e:
            logger.error('Failed to send response: {0}'.format(e))
