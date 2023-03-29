from abc import ABC, abstractmethod
from typing import Mapping, Optional, List, Any
import torch
from transformers import pipeline
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms.base import LLM
from llama_index import GPTSimpleVectorIndex, LangchainEmbedding, LLMPredictor


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_answer(self, context: str, question: str) -> str:
        pass


class CustomLLM(LLM):
    def __init__(self, model_name: str):
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=0,
            model_kwargs={"torch_dtype":torch.bfloat16}
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": "empty"}

    @property
    def _llm_type(self) -> str:
        return "custom"


class LangChainModel(Model):
    def __init__(
        self,
        hf_api_key: str,
        question_answering_model_id: str,
        embedding_model_id: str,
        run_localy: bool = False
    ):
        super().__init__()
        if run_localy:
            self.model = CustomLLM(question_answering_model_id)
        else:
            self.model = HuggingFaceHub(
                repo_id=question_answering_model_id,
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

        embedding_model = HuggingFaceHubEmbeddings(
            repo_id=embedding_model_id,
            huggingfacehub_api_token=hf_api_key
        )
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
