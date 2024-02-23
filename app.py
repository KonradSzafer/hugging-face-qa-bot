import threading

import gradio as gr

from qa_engine import logger, Config, QAEngine
from discord_bot import DiscordClient



config = Config()
qa_engine = QAEngine(config=config)


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
        config=config
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
