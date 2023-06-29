import uvicorn
from fastapi import FastAPI

from api.config import Config
from api.question_answering import QAModel
from api.logger import logger


config = Config()
app = FastAPI()
qa_model = QAModel(
    llm_model_id=config.question_answering_model_id,
    embedding_model_id=config.embedding_model_id,
    index_repo_id=config.index_repo_id,
    use_docs_for_context=config.use_docs_for_context,
    add_sources_to_response=config.add_sources_to_response,
    use_messages_for_context=config.use_messages_in_context,
    debug=config.debug
)


@app.get("/")
def get_answer(question: str, messgages_context: str):
    logger.info(
        f"Received request with question: {question}" \
        f"and context: {messgages_context}"
    )
    response = qa_model.get_answer(
        question=question,
        messages_context=messgages_context
    )
    return {
        "answer": response.get_answer(),
        "sources": response.get_sources_as_text()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
