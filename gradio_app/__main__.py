import gradio as gr

from qa_engine import logger, Config, QAEngine


config = Config()
model = QAEngine(
    llm_model_id=config.question_answering_model_id,
    embedding_model_id=config.embedding_model_id,
    index_repo_id=config.index_repo_id,
    prompt_template=config.prompt_template,
    use_docs_for_context=config.use_docs_for_context,
    add_sources_to_response=config.add_sources_to_response,
    use_messages_for_context=config.use_messages_in_context,
    debug=config.debug
)


if __name__ == '__main__':

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            context = ''.join(f'User: {msg} \nBot:{bot_msg}\n' for msg, bot_msg in chat_history)
            logger.info(f'Context: {context}')
            response = model.get_response(message, context)
            bot_message = response.get_answer() + response.get_sources_as_text() + '\n'
            chat_history.append((message, bot_message))
            return '', chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    demo.launch(share=True)
