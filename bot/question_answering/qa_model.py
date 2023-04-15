from abc import ABC, abstractmethod
from typing import Mapping, Optional, List
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from bot.logger import logger


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_answer(self, context: str, question: str) -> str:
        pass


class LangChainModel(Model):
    def __init__(
        self,
        llm_model_id: str,
        embedding_model_id: str,
        run_locally: bool = True,
        use_docs_for_context: bool = True,
        use_messages_for_context: bool = True,
        debug: bool = True
    ):
        super().__init__()
        self.use_docs_for_context = use_docs_for_context
        self.use_messages_for_context = use_messages_for_context
        self.debug = debug
        self.model_kwargs = {
            'min_length': 100,
            'max_length': 2000,
            'temperature': 0.1,
        }

        if run_locally:
            logger.info('running models locally')
            llm_model = HuggingFacePipeline.from_model_id(
                model_id=llm_model_id,
                task='text2text-generation',
                model_kwargs=self.model_kwargs
            )
            embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_id)
        else:
            logger.info('running models on huggingface hub')
            llm_model = HuggingFaceHub(
                repo_id=llm_model_id,
                model_kwargs=self.model_kwargs
            )
            embedding_model = HuggingFaceHubEmbeddings(repo_id=embedding_model_id)

        template = 'BEGINNING OF CONTEXT {context} END OF CONTEXT \n QUESTION: {question}'
        prompt = PromptTemplate(
            template=template,
            input_variables=['question', 'context']
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=llm_model)
        self.knowledge_index = FAISS.load_local("./index", embedding_model)


    def get_answer(self, question: str, messages_context: str = '') -> str:
        context = ''
        relevant_docs = ''
        if self.use_messages_for_context:
            context += 'MESSAGES CONTEXT:\n' + messages_context
        if self.use_docs_for_context:
            relevant_docs = self.knowledge_index.similarity_search(
                query=question,
                k=1
            )
            context += '\nRETRIEVED DOCUMENTS THAT MAY CONTAIN INFO RELEVANT TO QUESTION:'
            context += "".join([doc.page_content for doc in relevant_docs])
        response = self.llm_chain.run(question=question, context=context)
        if self.debug:
            sep = '\n' + '-' * 50
            logger.info(sep)
            logger.info(f'messages_contex: {messages_context} {sep}')
            logger.info(f'relevant_docs: {relevant_docs} {sep}')
            logger.info(f'context: {context} {sep}')
            logger.info(f'question: {question} {sep}')
            logger.info(f'response: {response} {sep}')
        return response
