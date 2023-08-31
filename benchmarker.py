import gradio as gr
from dotenv import load_dotenv
from api.config import Config
from api.logger import logger
from api.question_answering import QAModel
import time


load_dotenv(dotenv_path='config/api/.env')

def time_question(question: str, model: QAModel, context: str = "") -> str:
    t0 = time.perf_counter()
    response = model.get_response(question, context)
    t1 = time.perf_counter()
    return str(t1 - t0) + "\n" + response.get_answer() + response.get_sources_as_text() + "\n"

def main():

    config = Config()
    model = QAModel(
        llm_model_id=config.question_answering_model_id,
        embedding_model_id=config.embedding_model_id,
        index_repo_id=config.index_repo_id,
        use_docs_for_context=config.use_docs_for_context,
        add_sources_to_response=config.add_sources_to_response,
        use_messages_for_context=config.use_messages_in_context,
        debug=config.debug
    )

    with open("questions.txt", "r") as f:
        questions = f.readlines()

    with open("bench_answers.txt", "w") as f:
        for question in questions:
            output = ""
            question = question.strip()
            if "followup:" in question:
                q1, q2 = question.split("followup:")
                output += time_question(q1, model)
                output += time_question(q2, model, context=output)
            else:
                output += time_question(question, model)
            f.write(output + "\n====================================================\n")

if __name__ == "__main__":
    main()
