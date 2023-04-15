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
        model_id: str,
    ):
        super().__init__()
        self.model = HuggingFaceHub(
            repo_id=model_id,
            model_kwargs={
                'temperature': 0.1,
            }, 
        )
        print(model_id)
        model_name = "hkunlp/instructor-large"
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embed_instruction = "Represent the Hugging Face library documentation"
        query_instruction = "Query the most relevant piece of information from the Hugging Face documentation"
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.template = 'BEGINNING OF CONTEXT {context} END OF CONTEXT \n QUESTION: {question}'
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=['question', 'context']
        )
        
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.model)
        self.knowledge_index = FAISS.load_local("./index", embedding_model)


    def get_answer(self, context: str, question: str) -> str:
        # get 3 most relevant documents
        relevant_docs = self.knowledge_index.similarity_search(
            query=context,
            k=1
        )
        # add them to the context
        context = context + '\nRETRIEVED DOCUMENTS THAT MAY CONTAIN INFO RELEVANT TO QUESTION:' + "".join([doc.page_content for doc in relevant_docs])
        print(context)
        resp=self.llm_chain.run(question=question, context="")
        print(f"RESPOSNE: {resp}")
        return resp