from typing import List, Optional
import discord
from bot.logger import logger
from bot.question_answering.response import Response
from bot.question_answering import LangChainModel
from discord import app_commands
from bot.discord_client.utils import split_text_into_chunks


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
        enable_commands: bool = True,
        intents: Optional[discord.Intents] = None
    ):
        logger.info('Initializing Discord client...')
        intents = discord.Intents.all()
        if intents is None:
            intents = discord.Intents.default()
        intents.members = True
        super().__init__(intents=intents, command_prefix='/')

        assert num_last_messages >= 1, \
            'The number of last messages in context should be at least 1'

        self.model: LangChainModel = model
        self.num_last_messages: int = num_last_messages
        self.use_names_in_context: bool = use_names_in_context
        self.enable_commands: bool = enable_commands
        self.min_messgae_len: int = 1800
        self.max_message_len: int = 2000
        self.tree = app_commands.CommandTree(self)

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


    async def send_message(self, message, response: Response):
        chunks = split_text_into_chunks(
            text=response.get_response(),
            split_characters=[". ", ", ", "\n"],
            min_size=self.min_messgae_len,
            max_size=self.max_message_len
        )
        for chunk in chunks:
            await message.channel.send(chunk)
        await message.channel.send(response.get_sources_as_text())

    async def setup_hook(self) -> None:
        self.tree.add_command(hugginghelp)
        await self.tree.sync()
    
    async def on_guild_join(self, guild: discord.Guild) -> None:
        await self.tree.sync(guild=guild)

    async def generate_answer(self, interaction: discord.Interaction, question_str: str) -> None:
        """
        Generate an answer to a question.

        Args:
            interaction (discord.Interaction): The interaction that triggered the command.
            question_str (str): The question string.
        """
        context = '\n'.join(await self.get_last_messages(interaction)) # use interaction to get the last messages
        logger.info(f'Received question: {question_str}')
        response = self.model.get_answer(question_str, context)
        logger.info('Generated response: {0}'.format(response.get_response()))

        await interaction.edit_original_response(content=f"{interaction.user.mention} asked: **{question_str}** \n{response}")
        logger.info("Response sent")


@app_commands.command(name="hugginghelp", description="Help me with Hugging Face")
async def hugginghelp(interaction: discord.Interaction, question: str) -> None:
    """
    Command to generate answers to questions.

    Args:
        interaction (discord.Interaction): The interaction that triggered the command.
        prompt (str): The question to be answered.

    """
    if question != "":
        await interaction.response.send_message("Thinking... 🤔")
        interaction.client.loop.create_task(interaction.client.generate_answer(interaction, question))
    else:
        await interaction.response.send_message("Please provide a question")
