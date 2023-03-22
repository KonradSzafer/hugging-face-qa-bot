from typing import List
import discord
from bot.logger import logger
from bot.question_answering import Model


class DiscordClient(discord.Client):
    def __init__(
        self,
        model: Model,
        num_last_messages: int = 5,
        use_names_in_context: bool = True
    ):
        logger.info('Initializing Discord client...')
        intents = discord.Intents.all()
        intents.message_content = True
        super().__init__(intents=intents, command_prefix='!')
        self.model = model
        self.num_last_messages: int = num_last_messages
        self.use_names_in_context: bool = use_names_in_context

    def _set_initial_prompt(self) -> None:
        name = str(self.user).split('#')[0]
        self.initial_prompt: str = \
            f'your name is: {name}\n' \
            f'previous conversation messages:'

    async def on_ready(self):
        logger.info('Successfully logged in as: {0.user}'.format(self))
        self._set_initial_prompt()

    async def on_message(self, message):
        if message.author == self.user:
            return

        last_messages: List[str] = []
        async for msg in message.channel.history(
            limit=self.num_last_messages):
            if self.use_names_in_context:
                last_messages.append(f'{msg.author}: {msg.content}')
            else:
                last_messages.append(msg.content)
        last_messages.reverse()
        last_messages.pop() # remove last message from context
        context = self.initial_prompt + '\n'
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
