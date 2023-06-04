import os
import subprocess
from typing import Mapping, Optional, List, Any
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from llama_cpp import Llama

from bot.logger import logger
from bot.question_answering.response import Response


class LocalBinaryModel(LLM):
    """
    Local Binary Model class, used for loading a locally stored Llama model.

    Args:
        model_id (str): The ID of the model to be loaded.

    Attributes:
        model_path (str): The path to the model to be loaded.
        llm (Llama): The Llama object containing the loaded model.

    Raises:
        ValueError: If the model_path does not exist.

    """
    model_path: str = None
    llm: Llama = None

    def __init__(self, model_id: str = None):
        super().__init__()
        self.model_path = f'bot/question_answering/{model_id}'
        if not os.path.exists(self.model_path):
            raise ValueError(f'{self.model_path} does not exist')
        self.llm = Llama(model_path=self.model_path, n_ctx=4096)


    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = f'Q: {prompt} A: '
        output = self.llm(
            prompt,
            max_tokens=1024,
            stop=['Q:'],
            echo=False
        )
        output_text = output['choices'][0]['text']
        return output_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_path}

    @property
    def _llm_type(self) -> str:
        return self.model_path


class LangChainModel():
    """
    LangChainModel class, used for generating answers to questions.

    Args:
        llm_model_id (str): The ID of the LLM model to be used.
        embedding_model_id (str): The ID of the embedding model to be used.
        index_name (str): The name of the FAISS index to be used.
        run_locally (bool, optional): Whether to run the models locally or on the Hugging Face hub. Defaults to True.
        use_docs_for_context (bool, optional): Whether to use relevant documents as context for generating answers.
        Defaults to True.
        use_messages_for_context (bool, optional): Whether to use previous messages as context for generating answers.
        Defaults to True.
        debug (bool, optional): Whether to log debug information. Defaults to False.

    Attributes:
        use_docs_for_context (bool): Whether to use relevant documents as context for generating answers.
        use_messages_for_context (bool): Whether to use previous messages as context for generating answers.
        debug (bool): Whether to log debug information.
        model_kwargs (Dict[str, Any]): The model keyword arguments to be used.
        llm_model (Union[LocalBinaryModel, HuggingFacePipeline, HuggingFaceHub]): The LLM model to be used.
        embedding_model (Union[HuggingFaceInstructEmbeddings, HuggingFaceHubEmbeddings]): The embedding model to be used.
        prompt_template (PromptTemplate): The prompt template to be used.
        llm_chain (LLMChain): The LLM chain to be used.
        knowledge_index (FAISS): The FAISS index to be used.

    """
    def __init__(
        self,
        llm_model_id: str,
        embedding_model_id: str,
        index_name: str,
        run_locally: bool = True,
        use_docs_for_context: bool = True,
        add_sources_to_response: bool = True,
        use_messages_for_context: bool = True,
        debug: bool = False
    ):
        super().__init__()
        self.use_docs_for_context = use_docs_for_context
        self.add_sources_to_response = add_sources_to_response
        self.use_messages_for_context = use_messages_for_context
        self.debug = debug
        self.model_kwargs = {
            'min_length': 100,
            'max_length': 2000,
            'temperature': 0.1,
        }

        if run_locally:
            logger.info('running models locally')
            if 'local_models/' in llm_model_id:
                self.llm_model = LocalBinaryModel(
                    model_id=llm_model_id
                )
            else:
                self.llm_model = HuggingFacePipeline.from_model_id(
                    model_id=llm_model_id,
                    task='text2text-generation',
                    model_kwargs=self.model_kwargs
                )
            embed_instruction = "Represent the Hugging Face library documentation"
            query_instruction = "Query the most relevant piece of information from the Hugging Face documentation"
            embedding_model = HuggingFaceInstructEmbeddings(
                model_name=embedding_model_id,
                embed_instruction=embed_instruction,
                query_instruction=query_instruction
            )
        else:
            logger.info('running models on huggingface hub')
            self.llm_model = HuggingFaceHub(
                repo_id=llm_model_id,
                model_kwargs=self.model_kwargs
            )
            embedding_model = HuggingFaceHubEmbeddings(repo_id=embedding_model_id)

        prompt_template = \
            "### Instruction:\n" \
            "Give an answer that contains all the necessary information for the question.\n" \
            "If the context contains necessary information to answer question, use it to generate an appropriate response.\n" \
            "{context}\n### Input:\n{question}\n### Response:"

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=['question', 'context']
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm_model)
        self.knowledge_index = FAISS.load_local(f"./{index_name}", embedding_model)


    def get_answer(self, question: str, messages_context: str = '') -> Response:
        """
        Generate an answer to the specified question.

        Args:
            question (str): The question to be answered.
            messages_context (str, optional): The context to be used for generating the answer. Defaults to ''.

        Returns:
            response (Response): The Response object containing the generated answer and the sources of information 
            used to generate the response.
        """

        response = Response()
        context = 'Give an answer that contains all the necessary information for the question.\n'
        relevant_docs = ''
        if self.use_messages_for_context and messages_context:
            messages_context = f'\nPrevious questions and answers:\n{messages_context}'
            context += messages_context
        if self.use_docs_for_context:
            relevant_docs = self.knowledge_index.similarity_search(
                query=messages_context+question,
                k=3
            )
            context += '\nExtracted documents:\n'
            context += "".join([doc.page_content for doc in relevant_docs])
            metadata = [doc.metadata for doc in relevant_docs]
            response.set_sources(sources=[str(m['source']) for m in metadata])

        answer = self.llm_chain.run(question=question, context=context)
        response.set_response(answer)

        if self.debug:
            sep = '\n' + '-' * 100
            logger.info(sep)
            logger.info(f'messages_contex: {messages_context} {sep}')
            logger.info(f'relevant_docs: {relevant_docs} {sep}')
            sources_str = '\n'.join(response.get_sources())
            logger.info(f"sources:\n{sources_str}")
            logger.info(f'context len: {len(context)}')
            logger.info(f'context: {context} {sep}')
            logger.info(f'question: {question} {sep}')
            logger.info(f'response: {response.get_response()} {sep}')
        return response
