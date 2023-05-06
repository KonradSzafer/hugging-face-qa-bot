import os
import subprocess
from typing import Mapping, Optional, List, Any
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from llama_cpp import Llama
from bot.logger import logger


class LocalBinaryModel(LLM):
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


    def get_answer(self, question: str, messages_context: str = '') -> str:
        context = 'Give an answer that contains all the necessary information for the question.\n'
        relevant_docs = ''
        sources = []
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
            sources += list(set([str(m['source']) for m in metadata]))

        response = self.llm_chain.run(question=question, context=context)
        if self.add_sources_to_response:
            response += '\n\nSources:'
            for i, (source) in enumerate(sources):
                logger.info(source)
                response += f'\n [{i+1}] {source}'

        if self.debug:
            sep = '\n' + '-' * 100
            logger.info(sep)
            logger.info(f'messages_contex: {messages_context} {sep}')
            logger.info(f'relevant_docs: {relevant_docs} {sep}')
            sources_str = '\n'.join(sources)
            logger.info(f"sources:\n{sources_str}")
            logger.info(f'context len: {len(context)}')
            logger.info(f'context: {context} {sep}')
            logger.info(f'question: {question} {sep}')
            logger.info(f'response: {response} {sep}')
        return response
