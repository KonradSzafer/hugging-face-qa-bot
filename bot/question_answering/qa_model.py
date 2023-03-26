from abc import ABC, abstractmethod
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms.base import LLM
from llama_index import GPTSimpleVectorIndex, LangchainEmbedding, LLMPredictor
from typing import Mapping, Optional, List


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_answer(self, context: str, question: str) -> str:
        pass


class LangChainModel(Model):
    def __init__(
        self,
        hf_api_key: str,
        model_id: str,
    ):
        super().__init__()
        self.model = HuggingFaceHub(
            repo_id=model_id,
            model_kwargs={
                'max_length': 1000,
                'length_penalty': 2,
                'num_beams': 16,
                'no_repeat_ngram_size': 2,
                'temperature': 0.8,
                'top_k': 256,
                'top_p': 0.8
            }, 
            huggingfacehub_api_token=hf_api_key
        )
        embedding_model = HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token=hf_api_key)
        self.template = 'Question: {question} \nContext: {context}'
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=['question', 'context']
        )
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.model)
        self.knowledge_index = GPTSimpleVectorIndex.load_from_disk(
            "index.json",
            embed_model=LangchainEmbedding(embedding_model),
            llm_predictor=LLMPredictor(self.model)
        )


    def get_answer(self, context: str, question: str) -> str:
        doc_extracted = self.knowledge_index.query(question).response
        context += doc_extracted
        return self.llm_chain.run(
            question=question,
            context=context
        )
