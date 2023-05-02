from typing import List
import discord
from bot.logger import logger
from bot.question_answering import LangChainModel


class DiscordClient(discord.Client):
    """
    Discord Client class, used for interacting with a Discord server.

    Args:
        model (LangChainModel): The LangChainModel to be used for generating answers.
        num_last_messages (int, optional): The number of previous messages to use as context for generating answers.
        Defaults to 5.
        use_names_in_context (bool, optional): Whether to include user names in the message context. Defaults to True.
        enable_commands (bool, optional): Whether to enable commands for the bot. Defaults to True.

    Attributes:
        model (LangChainModel): The LangChainModel to be used for generating answers.
        num_last_messages (int): The number of previous messages to use as context for generating answers.
        use_names_in_context (bool): Whether to include user names in the message context.
        enable_commands (bool): Whether to enable commands for the bot.
        max_message_len (int): The maximum length of a message.
        system_prompt (str): The system prompt to be used.

    """
    def __init__(
        self,
        model: LangChainModel,
        num_last_messages: int = 5,
        use_names_in_context: bool = True,
        enable_commands: bool = True
    ):
        logger.info('Initializing Discord client...')
        intents = discord.Intents.all()
        intents.message_content = True
        super().__init__(intents=intents, command_prefix='!')

        assert num_last_messages >= 1, \
            'The number of last messages in context should be at least 1'

        self.model: LangChainModel = model
        self.num_last_messages: int = num_last_messages
        self.use_names_in_context: bool = use_names_in_context
        self.enable_commands: bool = enable_commands
        self.max_message_len: int = 2000

    def _set_system_prompt(self) -> None:
        name = str(self.user).split('#')[0]
        self.system_prompt: str = '' #\
            # f'your name is: {name}\n'

    async def on_ready(self):
        """
        Callback function to be called when the client is ready.
        """
        logger.info('Successfully logged in as: {0.user}'.format(self))
        await self.change_presence(activity=discord.Game(name='Chatting...'))
        self._set_system_prompt()

    async def on_message(self, message):
        """
        Callback function to be called when a message is received.

        Args:
            message (discord.Message): The received message.
        """
        if message.author == self.user:
            return
        if self.enable_commands and message.content.startswith('!'):
            if message.content == '!clear':
                await message.channel.purge()
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
        context = self.system_prompt + '\n'
        context += '\n'.join(last_messages)

        logger.info('Received message: {0.content}'.format(message))
        response = self.model.get_answer(message.content, context)
        logger.info('Sending response: {0}'.format(response))
        try:
            if len(response) > self.max_message_len:
                logger.warning(
                    f'generated response was to long: {len(response)} characters ' \
                    f'truncating to {self.max_message_len} characters'
                )
                response = response[:self.max_message_len]
            await message.channel.send(response)
        except Exception as e:
            logger.error('Failed to send response: {0}'.format(e))
