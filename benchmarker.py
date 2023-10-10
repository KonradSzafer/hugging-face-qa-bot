import gradio as gr
from dotenv import load_dotenv
from api.config import Config
from api.logger import logger
from api.question_answering import QAModel
import time


load_dotenv(dotenv_path='config/api/.env')

config = Config()
model = QAModel(
    llm_model_id=config.question_answering_model_id,
    embedding_model_id=config.embedding_model_id,
    index_repo_id=config.index_repo_id,
    prompt_template=config.prompt_template,
    use_docs_for_context=config.use_docs_for_context,
    add_sources_to_response=config.add_sources_to_response,
    use_messages_for_context=config.use_messages_in_context,
    debug=config.debug
)

QUESTIONS_FILENAME = 'data/benchmark/questions.json'
ANSWERS_FILENAME = 'data/benchmark/answers.json'


def main():
    benchmark_name = \
        f'model: {config.question_answering_model_id}' \
        f'index: {config.index_repo_id}'

    wandb.init(
        project='HF-Docs-QA',
        name=f'model: {config.question_answering_model_id}',
        mode='run', # run/disabled
        config=config.asdict()
    )
    # log config to wandb

    with open(QUESTIONS_FILENAME, 'r') as f: # json
        questions = f.readlines()

    with open(ANSWERS_FILENAME, 'w') as f:
        for q in questions:
            question = q['question']
            messages_contex = q['messages_context']

            t_start = time.perf_counter()
            response = model.get_response(
                question=question, 
                messages_context=messages_context
            )
            t_end = time.perf_counter()
            # write to json
            {
                "answer": response.get_answer(),
                "sources": response.get_sources_as_text(),
                'time': t_end - t_start
            }


if __name__ == '__main__':
    main()
