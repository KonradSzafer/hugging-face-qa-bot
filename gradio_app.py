import gradio as gr
from bot.config import Config
from bot.logger import logger
from bot.question_answering import LangChainModel

config = Config()
model = LangChainModel(
    llm_model_id=config.question_answering_model_id,
    embedding_model_id=config.embedding_model_id,
    index_name=config.index_name,
    use_docs_for_context=config.use_docs_in_context,
    add_sources_to_response=config.add_sources_to_response,
    use_messages_for_context=config.use_messages_in_context,
    debug=config.debug
)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        context = "".join(f"User: {msg} \nBot:{bot_msg}\n" for msg, bot_msg in chat_history)
        logger.info(f"Context: {context}")
        answer = model.get_answer(message, context)
        bot_message = answer.get_response() + answer.get_sources_as_text() + "\n"
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()