import os
from qa_engine import Config, QAEngine

os.environ['QUESTION_ANSWERING_MODEL_ID'] = 'mock'
os.environ['EMBEDDING_MODEL_ID'] = 'hkunlp/instructor-large'
os.environ['INDEX_REPO_ID'] = 'KonradSzafer/index-instructor-large-812-m512-all_repos_above_50_stars'
os.environ['PROMPT_TEMPLATE_NAME'] = 'llama2'


def test_qa_engine():
    config = Config()
    qa_engine = QAEngine(config=config)
    response = qa_engine.get_response(
        question='What is the capital of Poland?',
    )
    assert response.get_answer() == 'Warsaw'
