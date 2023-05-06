from bot.config import Config
from bot.logger import logger
from bot.discord_client import DiscordClient
from bot.question_answering import LangChainModel


def main():
    logger.info('Starting Application...')
    config = Config()
    config.log()
    model = LangChainModel(
        llm_model_id=config.question_answering_model_id,
        embedding_model_id=config.embedding_model_id,
        index_name=config.index_name,
        run_locally=config.run_locally,
        use_docs_for_context=config.use_docs_in_context,
        add_sources_to_response=config.add_sources_to_response,
        use_messages_for_context=config.use_messages_in_context,
        debug=False
    )
    client = DiscordClient(
        model=model,
        num_last_messages=config.num_last_messages,
        use_names_in_context=config.use_names_in_context,
        enable_commands=config.enable_commands
    )
    client.run(config.discord_token)

if __name__ == '__main__':
    main()
