import json
import requests
from urllib.parse import quote
import discord
from typing import List

from bot.logger import logger
from bot.discord_client.utils import split_text_into_chunks


class DiscordClient(discord.Client):
    """
    Discord Client class, used for interacting with a Discord server.

    Args:
        qa_service_url (str): The URL of the question answering service.
        num_last_messages (int, optional): The number of previous messages to use as context for generating answers.
        Defaults to 5.
        use_names_in_context (bool, optional): Whether to include user names in the message context. Defaults to True.
        enable_commands (bool, optional): Whether to enable commands for the bot. Defaults to True.

    Attributes:
        qa_service_url (str): The URL of the question answering service.
        num_last_messages (int): The number of previous messages to use as context for generating answers.
        use_names_in_context (bool): Whether to include user names in the message context.
        enable_commands (bool): Whether to enable commands for the bot.
        max_message_len (int): The maximum length of a message.
        system_prompt (str): The system prompt to be used.

    """
    def __init__(
        self,
        qa_service_url: str,
        num_last_messages: int = 5,
        use_names_in_context: bool = True,
        enable_commands: bool = True,
        debug: bool = False
    ):
        logger.info('Initializing Discord client...')
        intents = discord.Intents.all()
        intents.message_content = True
        super().__init__(intents=intents, command_prefix='!')

        assert num_last_messages >= 1, \
            'The number of last messages in context should be at least 1'

        self.qa_service_url: str = qa_service_url
        self.num_last_messages: int = num_last_messages
        self.use_names_in_context: bool = use_names_in_context
        self.enable_commands: bool = enable_commands
        self.debug: bool = debug
        self.min_messgae_len: int = 1800
        self.max_message_len: int = 2000


    async def on_ready(self):
        """
        Callback function to be called when the client is ready.
        """
        logger.info('Successfully logged in as: {0.user}'.format(self))
        await self.change_presence(activity=discord.Game(name='Chatting...'))


    async def get_last_messages(self, message) -> List[str]:
        """
        Method to fetch recent messages from a message's channel.

        Args:
            message (Message): The discord Message object used to identify the channel.

        Returns:
            List[str]: Reversed list of recent messages from the channel,
            excluding the input message. Messages may be prefixed with the author's name 
            if `self.use_names_in_context` is True.
        """
        last_messages: List[str] = []
        async for msg in message.channel.history(
            limit=self.num_last_messages):
            if self.use_names_in_context:
                last_messages.append(f'{msg.author}: {msg.content}')
            else:
                last_messages.append(msg.content)
        last_messages.reverse()
        last_messages.pop() # remove last message from context
        return last_messages


    async def send_message(self, message, answer: str, sources: str):
        chunks = split_text_into_chunks(
            text=answer,
            split_characters=[". ", ", ", "\n"],
            min_size=self.min_messgae_len,
            max_size=self.max_message_len
        )
        for chunk in chunks:
            await message.channel.send(chunk)
        await message.channel.send(sources)


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

        last_messages = await self.get_last_messages(message)
        context = '\n'.join(last_messages)

        logger.info('Received message: {0.content}'.format(message))
        question_encoded = quote(message.content, safe='')
        context_encoded = quote(context, safe='')
        url = \
            f'{self.qa_service_url}/' \
            f'?question={question_encoded}' \
            f'?&messgages_context={context_encoded}'
        response = requests.get(url)
        response.raise_for_status()
        response = json.loads(response.content)

        logger.info('Sending response: {0}'.format(response))
        try:
            await self.send_message(message, response['answer'], response['sources'])
        except Exception as e:
            logger.error('Failed to send response: {0}'.format(e))
