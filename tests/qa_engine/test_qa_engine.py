import os
from qa_engine import Config, QAEngine


def test_qa_engine():
    config = Config()
    qa_engine = QAEngine(config=config)
    response = qa_engine.get_response(
        question='What is the capital of Poland?',
    )
    assert response.get_answer() == 'Warsaw'
