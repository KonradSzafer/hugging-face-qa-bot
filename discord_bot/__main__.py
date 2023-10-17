from qa_engine import logger, Config, QAEngine
from discord_bot.client import DiscordClient


config = Config()
qa_engine = QAEngine(
    llm_model_id=config.question_answering_model_id,
    embedding_model_id=config.embedding_model_id,
    index_repo_id=config.index_repo_id,
    prompt_template=config.prompt_template,
    use_docs_for_context=config.use_docs_for_context,
    num_relevant_docs=config.num_relevant_docs,
    add_sources_to_response=config.add_sources_to_response,
    use_messages_for_context=config.use_messages_in_context,
    debug=config.debug
)
client = DiscordClient(
    qa_engine=qa_engine,
    num_last_messages=config.num_last_messages,
    use_names_in_context=config.use_names_in_context,
    enable_commands=config.enable_commands,
    debug=config.debug
)


if __name__ == '__main__':
    logger.info('Starting Application...')
    client.run(config.discord_token)
