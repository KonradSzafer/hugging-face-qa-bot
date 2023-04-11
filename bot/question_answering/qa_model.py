from abc import ABC, abstractmethod
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings
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
        model_name = "hkunlp/instructor-large"
        embed_instruction = "Represent the Hugging Face library documentation"
        query_instruction = "Query the most relevant piece of information from the Hugging Face documentation"
        embedding_model = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            embed_instruction=embed_instruction,
            query_instruction=query_instruction,
        )
        self.template = 'Context: {context} \n Question: {question}'
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=['question', 'context']
        )
        
        self.llm_chain = LLMChain(prompt=self.prompt, llm=OpenAI(temperature=0.1))
        self.knowledge_index = FAISS.load_local("./index", embedding_model)


    def get_answer(self, context: str, question: str) -> str:
        # get 3 most relevant documents
        relevant_docs = self.knowledge_index.similarity_search(
            query=context,
            k=7
        )
        # add them to the context
        context = context + '\n Retrieved document info (format it and make output beautiful):' + "".join([doc.page_content for doc in relevant_docs])
        print(context)
        return self.llm_chain.run(
            question=question,
            context=context
        )