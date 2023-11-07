import time
import json

import wandb
import gradio as gr

from qa_engine import logger, Config, QAEngine


QUESTIONS_FILENAME = 'benchmark/questions.json'

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


def main():
    filtered_config = config.asdict()
    disallowed_config_keys = [
        "DISCORD_TOKEN", "NUM_LAST_MESSAGES", "USE_NAMES_IN_CONTEXT",
        "ENABLE_COMMANDS", "APP_MODE", "DEBUG"
    ]
    for key in disallowed_config_keys:
        filtered_config.pop(key, None)

    wandb.init(
        project='HF-Docs-QA',
        entity='hf-qa-bot',
        name=f'{config.question_answering_model_id} - {config.embedding_model_id} - {config.index_repo_id}',
        mode='run', # run/disabled
        config=filtered_config
    )

    with open(QUESTIONS_FILENAME, 'r') as f:
        questions = json.load(f)

    table = wandb.Table(
        columns=[
            "id", "question", "messages_context", "answer", "sources", "time"
        ]
    )
    for i, q in enumerate(questions):
        logger.info(f"Question {i+1}/{len(questions)}")

        question = q['question']
        messages_context = q['messages_context']

        time_start = time.perf_counter()
        response = qa_engine.get_response(
            question=question, 
            messages_context=messages_context
        )
        time_end = time.perf_counter()

        table.add_data(
            i,
            question,
            messages_context,
            response.get_answer(),
            response.get_sources_as_text(),
            time_end - time_start
        )

    wandb.log({"answers": table})
    wandb.finish()


if __name__ == '__main__':
    main()
