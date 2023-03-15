from typing import List
import discord
from bot.logger import logger


class DiscordClient(discord.Client):
    def __init__(self, model):
        logger.info('Initializing Discord client...')
        intents = discord.Intents.all()
        intents.message_content = True
        super().__init__(intents=intents, command_prefix='!')
        self.model = model
        self.n_last_messages: int = 5
        self.initial_prompt: str = ''

    async def on_ready(self):
        logger.info('Successfully logged in as: {0.user}'.format(self))

    async def on_message(self, message):
        if message.author == self.user:
            return

        context = self.initial_prompt + '\n'
        last_messages: List[str] = []
        async for msg in message.channel.history(
            limit=self.n_last_messages):
            last_messages.append(msg.content)
        last_messages.reverse()
        last_messages.pop() # remove last message from context
        context += '\n'.join(last_messages)

        logger.info('Received message: {0.content}'.format(message))
        response = self.model.get_answer(
            context,
            message.content
        )
        logger.info('Sending response: {0}'.format(response))
        try:
            await message.channel.send(response)
        except Exception as e:
            logger.error('Failed to send response: {0}'.format(e))
