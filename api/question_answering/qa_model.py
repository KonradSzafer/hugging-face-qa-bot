import os
import json
import requests
import subprocess
from urllib.parse import quote
from typing import Mapping, Optional, List, Any
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from llama_cpp import Llama

from api.logger import logger
from api.question_answering.response import Response


class LocalBinaryModel(LLM):
    model_id: str = None
    llm: Llama = None

    def __init__(self, model_id: str = None):
        super().__init__()
        model_path = f'api/question_answering/{model_id}'
        if not os.path.exists(model_path):
            raise ValueError(f'{model_path} does not exist')
        self.model_id = model_id
        self.llm = Llama(model_path=model_path, n_ctx=4096)

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
        return {"name_of_model": self.model_id}

    @property
    def _llm_type(self) -> str:
        return self.model_id


class TransformersPipelineModel(LLM):
    model_id: str = None
    pipeline: str = None

    def __init__(self, model_id: str = None):
        super().__init__()
        self.model_id = model_id

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_8bit=False,
            device_map="auto",
            resume_download=True,
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output_text = self.pipeline(prompt)[0]['generated_text']
        output_text = output_text.replace(prompt+'\n', '')
        return output_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_id}

    @property
    def _llm_type(self) -> str:
        return self.model_id


class APIServedModel(LLM):
    model_url: str = None

    def __init__(self, model_url: str = None):
        super().__init__()
        if model_url[-1] == '/':
            raise ValueError('URL should not end with a slash - "/"')
        self.model_url = model_url

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_encoded = quote(prompt, safe='')
        url = f'{self.model_url}/?prompt={prompt_encoded}'
        try:
            response = requests.get(url)
            response.raise_for_status() 
            output_text = json.loads(response.content)['output_text']
            return output_text
        except Exception as err:
            logger.error(f'Error: {err}')
            return 'Error: {err}'

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": f'model url: {self.model_url}'}

    @property
    def _llm_type(self) -> str:
        return 'api_model'



class QAModel():
    """
    QAModel class, used for generating answers to questions.

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

        if 'local_models/' in llm_model_id:
            logger.info('using local binary model')
            self.llm_model = LocalBinaryModel(
                model_id=llm_model_id
            )
        elif 'api_models/' in llm_model_id:
            logger.info('using api served model')
            self.llm_model = APIServedModel(
                model_url=llm_model_id.replace('api_models/', '')
            )
        else:
            logger.info('using transformers pipeline model')
            self.llm_model = TransformersPipelineModel(
                model_id=llm_model_id
            )

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

        if self.use_docs_for_context:
            embed_instruction = "Represent the Hugging Face library documentation"
            query_instruction = "Query the most relevant piece of information from the Hugging Face documentation"
            embedding_model = HuggingFaceInstructEmbeddings(
                model_name=embedding_model_id,
                embed_instruction=embed_instruction,
                query_instruction=query_instruction
            )
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
        response.set_answer(answer)

        if self.debug:
            sep = '\n' + '-' * 100
            logger.info(sep)
            logger.info(f'messages_contex: {messages_context} {sep}')
            logger.info(f'relevant_docs: {relevant_docs} {sep}')
            sources_str = '\n'.join(response.get_sources())
            logger.info(f"sources:\n{sources_str} {sep}")
            logger.info(f'context len: {len(context)} {sep}')
            logger.info(f'context: {context} {sep}')
            logger.info(f'question len: {len(question)}')
            logger.info(f'question: {question} {sep}')
            logger.info(f'response: {response.get_answer()} {sep}')
        return response
