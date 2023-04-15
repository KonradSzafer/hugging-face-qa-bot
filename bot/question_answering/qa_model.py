from abc import ABC, abstractmethod
from langchain import PromptTemplate, HuggingFaceHub, LLMChain 
from langchain.embeddings import HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from typing import Mapping, Optional, List
import InstructorEmbedding
import faiss


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
    ):
        super().__init__()
        model = HuggingFaceHub(
            repo_id=llm_model_id,
            model_kwargs={
                'temperature': 0.1,
            }, 
        )
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_id)
        template = 'BEGINNING OF CONTEXT {context} END OF CONTEXT \n QUESTION: {question}'
        prompt = PromptTemplate(
            template=template,
            input_variables=['question', 'context']
        )
        
        self.llm_chain = LLMChain(prompt=prompt, llm=model)
        self.knowledge_index = FAISS.load_local("./index", embedding_model)


    def get_answer(self, context: str, question: str) -> str:
        relevant_docs = self.knowledge_index.similarity_search(
            query=context,
            k=1
        )

        context += '\nRETRIEVED DOCUMENTS THAT MAY CONTAIN INFO RELEVANT TO QUESTION:'
        context += "".join([doc.page_content for doc in relevant_docs])
        response = self.llm_chain.run(question=question, context=context)
        return response