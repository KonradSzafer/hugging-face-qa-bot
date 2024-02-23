import json
import requests
from urllib.parse import quote
import discord
from typing import List

from qa_engine import logger, Config, QAEngine
from discord_bot.client.utils import split_text_into_chunks


class DiscordClient(discord.Client):
    """
    Discord Client class, used for interacting with a Discord server.
    """
    def __init__(
        self,
        qa_engine: QAEngine,
        config: Config,  
    ):  
        logger.info('Initializing Discord client...')
        intents = discord.Intents.all()
        intents.message_content = True
        super().__init__(intents=intents, command_prefix='!')

        self.qa_engine: QAEngine = qa_engine
        self.channel_ids: list[int] = DiscordClient._process_channel_ids(
            config.discord_channel_ids
        )
        self.num_last_messages: int = config.num_last_messages
        self.use_names_in_context: bool = config.use_names_in_context
        self.enable_commands: bool = config.enable_commands
        self.debug: bool = config.debug
        self.min_message_len: int = 1800
        self.max_message_len: int = 2000
        
        assert all([isinstance(id, int) for id in self.channel_ids]), \
            'All channel ids should be of type int'
        assert self.num_last_messages >= 1, \
            'The number of last messages in context should be at least 1'
        
    
    @staticmethod
    def _process_channel_ids(channel_ids) -> list[int]:
        if isinstance(channel_ids, str):
            return eval(channel_ids)
        elif isinstance(channel_ids, list):
            return channel_ids
        elif isinstance(channel_ids, int):
            return [channel_ids]
        else:
            return []


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
            split_characters=['. ', ', ', '\n'],
            min_size=self.min_message_len,
            max_size=self.max_message_len
        )
        for chunk in chunks:
            await message.channel.send(chunk)
        await message.channel.send(sources)


    async def on_message(self, message):

        if self.channel_ids and message.channel.id not in self.channel_ids:
            return
        
        if message.author == self.user:
            return
        
        """
        if self.enable_commands and message.content.startswith('!'):
            if message.content == '!clear':
                await message.channel.purge()
                return        
        """

        last_messages = await self.get_last_messages(message)
        context = '\n'.join(last_messages)

        logger.info('Received message: {0.content}'.format(message))
        response = self.qa_engine.get_response(
            question=message.content,
            messages_context=context
        )
        logger.info('Sending response: {0}'.format(response))
        try:
            await self.send_message(
                message,
                response.get_answer(),
                response.get_sources_as_text()
            )
        except Exception as e:
            logger.error('Failed to send response: {0}'.format(e))
