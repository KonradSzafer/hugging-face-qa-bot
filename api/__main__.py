import uvicorn
from fastapi import FastAPI

from qa_engine import logger, Config, QAEngine


config = Config()
app = FastAPI()
qa_engine = QAEngine(config=config)


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
