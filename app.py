import threading

import gradio as gr

from qa_engine import logger, Config, QAEngine
from discord_bot import DiscordClient



config = Config()
qa_engine = QAEngine(
    llm_model_id=config.question_answering_model_id,
    embedding_model_id=config.embedding_model_id,
    index_repo_id=config.index_repo_id,
    prompt_template=config.prompt_template,
    use_docs_for_context=config.use_docs_for_context,
    add_sources_to_response=config.add_sources_to_response,
    use_messages_for_context=config.use_messages_in_context,
    debug=config.debug
)


def gradio_interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            context = ''.join(f'User: {msg} \nBot:{bot_msg}\n' for msg, bot_msg in chat_history)
            logger.info(f'Context: {context}')
            response = qa_engine.get_response(message, context)
            bot_message = response.get_answer() + response.get_sources_as_text() + '\n'
            chat_history.append((message, bot_message))
            return '', chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    demo.launch(share=True)


def discord_bot_inference_thread():
    client = DiscordClient(
        qa_engine=qa_engine,
        num_last_messages=config.num_last_messages,
        use_names_in_context=config.use_names_in_context,
        enable_commands=config.enable_commands,
        debug=config.debug
    )
    client.run(config.discord_token)

def discord_bot():
    thread = threading.Thread(target=discord_bot_inference_thread)
    thread.start()
    with gr.Blocks() as demo:
        gr.Markdown(f'Discord bot is running.')
    demo.queue(concurrency_count=100)
    demo.queue(max_size=100)
    demo.launch()


if __name__ == '__main__':
    if config.app_mode == 'gradio':
        gradio_interface()
    elif config.app_mode == 'discord':
        discord_bot()
    else:
        raise ValueError(
            f'Invalid app mode: {config.app_mode}, ',
            f'set APP_MODE to "gradio" or "discord"'
        )
