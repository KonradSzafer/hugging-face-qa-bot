import os
import json
import requests
import subprocess
from typing import Mapping, Optional, Any

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from urllib.parse import quote
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from qa_engine import logger
from qa_engine.response import Response
from qa_engine.mocks import MockLocalBinaryModel


class LocalBinaryModel(LLM):
    model_id: str = None
    llm: None = None

    def __init__(self, model_id: str = None):
        super().__init__()
        # pip install llama_cpp_python==0.1.39
        from llama_cpp import Llama

        model_path = f'qa_engine/{model_id}'
        if not os.path.exists(model_path):
            raise ValueError(f'{model_path} does not exist')
        self.model_id = model_id
        self.llm = Llama(model_path=model_path, n_ctx=4096)

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        output = self.llm(
            prompt,
            max_tokens=1024,
            stop=['Q:'],
            echo=False
        )
        return output['choices'][0]['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'name_of_model': self.model_id}

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
            device_map='auto',
            resume_download=True,
        )
        self.pipeline = transformers.pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            min_new_tokens=64,
            max_new_tokens=800,
            temperature=0.5,
            do_sample=True,
        )

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        output_text = self.pipeline(prompt)[0]['generated_text']
        output_text = output_text.replace(prompt+'\n', '')
        return output_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'name_of_model': self.model_id}

    @property
    def _llm_type(self) -> str:
        return self.model_id


class APIServedModel(LLM):
    model_url: str = None
    debug: bool = None

    def __init__(self, model_url: str = None, debug: bool = None):
        super().__init__()
        if model_url[-1] == '/':
            raise ValueError('URL should not end with a slash - "/"')
        self.model_url = model_url
        self.debug = debug

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        prompt_encoded = quote(prompt, safe='')
        url = f'{self.model_url}/?prompt={prompt_encoded}'
        if self.debug:
            logger.info(f'URL: {url}')
        try:
            response = requests.get(url, timeout=1200, verify=False)
            response.raise_for_status() 
            return json.loads(response.content)['output_text']
        except Exception as err:
            logger.error(f'Error: {err}')
            return f'Error: {err}'

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'name_of_model': f'model url: {self.model_url}'}

    @property
    def _llm_type(self) -> str:
        return 'api_model'



class QAEngine():
    """
    QAEngine class, used for generating answers to questions.

    Args:
        llm_model_id (str): The ID of the LLM model to be used.
        embedding_model_id (str): The ID of the embedding model to be used.
        index_repo_id (str): The ID of the index repository to be used.
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
        index_repo_id: str,
        prompt_template: str,
        use_docs_for_context: bool = True,
        num_relevant_docs: int = 3,
        add_sources_to_response: bool = True,
        use_messages_for_context: bool = True,
        first_stage_docs: int = 50,
        debug: bool = False
    ):
        super().__init__()
        self.prompt_template = prompt_template
        self.use_docs_for_context = use_docs_for_context
        self.num_relevant_docs = num_relevant_docs
        self.add_sources_to_response = add_sources_to_response
        self.use_messages_for_context = use_messages_for_context
        self.first_stage_docs = first_stage_docs
        self.debug = debug

        if 'local_models/' in llm_model_id:
            logger.info('using local binary model')
            self.llm_model = LocalBinaryModel(
                model_id=llm_model_id
            )
        elif 'api_models/' in llm_model_id:
            logger.info('using api served model')
            self.llm_model = APIServedModel(
                model_url=llm_model_id.replace('api_models/', ''),
                debug=self.debug
            )
        elif llm_model_id == 'mock':
            logger.info('using mock model')
            self.llm_model = MockLocalBinaryModel()
        else:
            logger.info('using transformers pipeline model')
            self.llm_model = TransformersPipelineModel(
                model_id=llm_model_id
            )

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=['question', 'context']
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm_model)

        if self.use_docs_for_context:
            logger.info(f'Downloading {index_repo_id}')
            snapshot_download(
                repo_id=index_repo_id,
                allow_patterns=['*.faiss', '*.pkl'], 
                repo_type='dataset',
                local_dir='indexes/run/'
            )
            logger.info('Loading embedding model')
            embed_instruction = 'Represent the Hugging Face library documentation'
            query_instruction = 'Query the most relevant piece of information from the Hugging Face documentation'
            embedding_model = HuggingFaceInstructEmbeddings(
                model_name=embedding_model_id,
                embed_instruction=embed_instruction,
                query_instruction=query_instruction
            )
            logger.info('Loading index')
            self.knowledge_index = FAISS.load_local('./indexes/run/', embedding_model)
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


    @staticmethod
    def _preprocess_question(question: str) -> str:
        if question[-1] != '?':
            question += '?'
        return question


    @staticmethod
    def _postprocess_answer(answer: str) -> str:
        '''
        Preprocess the answer by removing unnecessary sequences and stop sequences.
        '''
        SEQUENCES_TO_REMOVE = [
            'Factually: ', 'Answer: ', '<<SYS>>', '<</SYS>>', '[INST]', '[/INST]'
        ]
        SEQUENCES_TO_STOP = [
            'User:', 'You:', 'Question:'
        ]
        for seq in SEQUENCES_TO_REMOVE:
            answer = answer.replace(seq, '')
        for seq in SEQUENCES_TO_STOP:
            if seq in answer:
                answer = answer[:answer.index(seq)]
        answer = answer.strip()
        return answer


    def get_response(self, question: str, messages_context: str = '') -> Response:
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
        context = ''
        relevant_docs = ''
        if self.use_messages_for_context and messages_context:
            messages_context = f'\nPrevious questions and answers:\n{messages_context}'
            context += messages_context
        if self.use_docs_for_context:
            logger.info('Retriving documents')
            # messages context is used for better retrival
            retrival_query = messages_context + question 
            relevant_docs = self.knowledge_index.similarity_search(
                query=retrival_query,
                k=self.first_stage_docs
            )
            cross_encoding_predictions = self.reranker.predict(
                [(retrival_query, doc.page_content) for doc in relevant_docs]
            )
            relevant_docs = [
                doc for _, doc in sorted(
                    zip(cross_encoding_predictions, relevant_docs),
                    reverse=True, key = lambda x: x[0]
                )
            ]
            relevant_docs = relevant_docs[:self.num_relevant_docs]
            context += '\nExtracted documents:\n'
            context += ''.join([doc.page_content for doc in relevant_docs])
            metadata = [doc.metadata for doc in relevant_docs]
            response.set_sources(sources=[str(m['source']) for m in metadata])

        logger.info('Running LLM chain')
        question_processed = QAEngine._preprocess_question(question)
        answer = self.llm_chain.run(question=question_processed, context=context)
        answer_postprocessed = QAEngine._postprocess_answer(answer)
        response.set_answer(answer_postprocessed)
        logger.info('Received answer')

        if self.debug:
            logger.info('\n' + '=' * 100)
            sep = '\n' + '-' * 100
            logger.info(f'question len: {len(question)} {sep}')
            logger.info(f'question: {question} {sep}')
            logger.info(f'answer len: {len(response.get_answer())} {sep}')
            logger.info(f'answer original: {answer} {sep}')
            logger.info(f'answer postprocessed: {response.get_answer()} {sep}')
            logger.info(f'{response.get_sources_as_text()} {sep}')
            logger.info(f'messages_contex: {messages_context} {sep}')
            logger.info(f'relevant_docs: {relevant_docs} {sep}')
            logger.info(f'context len: {len(context)} {sep}')
            logger.info(f'context: {context} {sep}')
        return response
