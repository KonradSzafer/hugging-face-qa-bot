import uvicorn
from fastapi import FastAPI

from qa_engine import logger, Config, QAEngine


config = Config()
app = FastAPI()
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


@app.get('/')
def get_answer(question: str, messages_context: str = ''):
    logger.info(
        f'Received request with question: {question}' \
        f'and context: {messages_context}'
    )
    response = qa_engine.get_response(
        question=question,
        messages_context=messages_context
    )
    return {
        'answer': response.get_answer(),
        'sources': response.get_sources_as_text()
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
