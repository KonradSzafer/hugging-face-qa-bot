from abc import ABC, abstractmethod
from typing import Mapping, Optional, List, Any
import json
import requests
import torch
from sentence_transformers import SentenceTransformer
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms.base import LLM
from llama_index import GPTSimpleVectorIndex, LLMPredictor, LangchainEmbedding, ServiceContext, PromptHelper
from llama_index.logger import LlamaLogger
from bot.logger import logger


class Model(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f'loaded question answerig model: {model_name}')

    @staticmethod
    def _prompt_formater(question: str, context: str) -> str:
        if len(context) == 0:
            return f"Question: {question}\n\nAnswer:"
        else:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    @abstractmethod
    def get_answer(self, question: str, context: str) -> str:
        pass


class FlanT5Local(Model):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # 'google/flan-t5-large'
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        from transformers import T5Tokenizer
        return T5Tokenizer.from_pretrained(self.model_name)

    def _load_model(self):
        from transformers import T5ForConditionalGeneration
        return T5ForConditionalGeneration.from_pretrained(self.model_name)

    def get_answer(self, question: str, context: str = '') -> str:
        prompt = self._prompt_formater(question, context)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


class RestAPIModel(Model):
    def __init__(self, model_name: str, hf_api_key: str):
        super().__init__(model_name)
        self.hf_api_key: str = hf_api_key
        self.url: str = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.hf_api_key}"}

    def get_answer(self, context: str, question: str = '') -> str:
        prompt = self._prompt_formater(question, context)
        data = json.dumps(prompt)
        response = requests.request("POST", self.url, headers=self.headers, data=data)
        response = json.loads(response.content.decode("utf-8"))
        return response[0]['generated_text']


class CustomLLM(LLM):
    model_name: str = None
    model: Model = None

    def __init__(
        self,
        model_name: str,
        hf_api_key: str,
        run_localy: bool,
    ):
        super().__init__()
        self.model_name = model_name
        if run_localy:
            self.model = FlanT5Local(model_name)
        else:
            self.model = RestAPIModel(model_name, hf_api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.model.get_answer(prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return self.model_name


class EmbeddingModel(SentenceTransformer):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.encode(prompt)

    def embed_query(self, query: str):
        return self.encode(query)


class LangChainModel():
    def __init__(
        self,
        question_answering_model_id: str,
        embedding_model_id: str,
        hf_api_key: str,
        run_localy: bool = True
    ):
        super().__init__()
        self.question_answering_model = CustomLLM(
            question_answering_model_id, hf_api_key, run_localy)
        # self.embedding_model = EmbeddingModel(
        #     embedding_model_id)
        # self.embedding_model = HuggingFaceEmbeddings()

        self.template = 'Question: {question} \nContext: {context}'
        self.prompt_template = PromptTemplate(
            template=self.template,
            input_variables=['question', 'context']
        )
        self.llm_chain = LLMChain(
            prompt=self.prompt_template,
            llm=self.question_answering_model
        )

        # max_input_size = 1096
        # num_output = 512
        # max_chunk_overlap = 50
        # service_context = ServiceContext(
        #     llm_predictor=LLMPredictor(llm=CustomLLM()),
        #     embed_model=LangchainEmbedding(embedding_model),
        #     prompt_helper=PromptHelper(max_input_size, num_output, max_chunk_overlap),
        #     node_parser=SimpleNodeParser(),
        #     llama_logger=LlamaLogger()
        # )
        # self.knowledge_index = GPTSimpleVectorIndex.load_from_disk(
        #     "index.json", service_context=service_context
        # )


    def get_answer(self, question: str, context: str = '') -> str:
        # docs_extracted = self.knowledge_index.query(question)
        # context += docs_extracted.source_nodes[0].node.get_text()
        response = self.llm_chain.run(
            question=question,
            context=context
        )
        return response
