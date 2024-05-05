import re
from typing import Mapping, Optional, Any

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from qa_engine import logger, Config
from qa_engine.response import Response
from qa_engine.mocks import MockLocalBinaryModel


class HuggingFaceModel:
    model_id: str = None
    min_new_tokens: int = None
    max_new_tokens: int = None
    temperature: float = None
    top_k: int = None
    top_p: float = None
    do_sample: bool = None
    tokenizer: transformers.PreTrainedTokenizer = None
    model: transformers.PreTrainedModel = None

    def __init__(self, config: Config):
        super().__init__()
        self.model_id = config.question_answering_model_id
        self.min_new_tokens = config.min_new_tokens
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.do_sample = config.do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        tokenized_prompt = self.tokenizer(
            self.tokenizer.bos_token + prompt, 
            return_tensors="pt"
        ).to(self.model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.model.generate(
            input_ids=tokenized_prompt.input_ids,
            attention_mask=tokenized_prompt.attention_mask,
            min_new_tokens=self.min_new_tokens,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators
        )
        response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
        decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
        return decoded_response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'name_of_model': self.model_id}

    @property
    def _llm_type(self) -> str:
        return self.model_id


class QAEngine():
    """
    QAEngine class, used for generating answers to questions.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.question_answering_model_id=config.question_answering_model_id
        self.embedding_model_id=config.embedding_model_id
        self.index_repo_id=config.index_repo_id
        self.prompt_template=config.prompt_template
        self.use_docs_for_context=config.use_docs_for_context
        self.num_relevant_docs=config.num_relevant_docs
        self.add_sources_to_response=config.add_sources_to_response
        self.use_messages_for_context=config.use_messages_in_context
        self.debug=config.debug        
        self.first_stage_docs: int = 50

        self.llm_model = self._get_model()

        if self.use_docs_for_context:
            logger.info(f'Downloading {self.index_repo_id}')
            snapshot_download(
                repo_id=self.index_repo_id,
                allow_patterns=['*.faiss', '*.pkl'], 
                repo_type='dataset',
                local_dir='indexes/run/'
            )
            logger.info('Loading embedding model')
            embed_instruction = 'Represent the Hugging Face library documentation'
            query_instruction = 'Query the most relevant piece of information from the Hugging Face documentation'
            embedding_model = HuggingFaceInstructEmbeddings(
                model_name=self.embedding_model_id,
                embed_instruction=embed_instruction,
                query_instruction=query_instruction
            )
            logger.info('Loading index')
            self.knowledge_index = FAISS.load_local('./indexes/run/', embedding_model)
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


    def _get_model(self):
        if self.question_answering_model_id == 'mock':
            logger.warn('using mock model')
            return MockLocalBinaryModel()
        else:
            logger.info('using transformers pipeline model')
            return HuggingFaceModel(self.config)
    
    @staticmethod
    def _preprocess_input(question: str, context: str) -> str:
        if '?' not in question:
            question += '?'
        
        # llama3 chatQA specific
        messages = [
            {"role": "user", "content": question}
        ]

        system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
        instruction = "Please give a full and complete answer for the question."

        for item in messages:
            if item['role'] == "user":
                ## only apply this instruction for the first user turn
                item['content'] = instruction + " " + item['content']
                break

        conversation = '\n\n'.join([
            "User: " + item["content"] if item["role"] == "user" else
            "Assistant: " + item["content"] for item in messages
        ]) + "\n\nAssistant:"

        inputs = system + "\n\n" + context + "\n\n" + conversation
        return inputs

    @staticmethod
    def _postprocess_answer(answer: str) -> str:
        '''
        Preprocess the answer by removing unnecessary sequences and stop sequences.
        '''
        SEQUENCES_TO_REMOVE = [
            'Factually: ', 'Answer: ', '<<SYS>>', '<</SYS>>', '[INST]', '[/INST]',
            '<context>', '</context>', '<question>', '</question>',
        ]
        SEQUENCES_TO_STOP = [
            'User:', 'You:', 'Question:'
        ]
        CHARS_TO_DEDUPLICATE = [
            '\n', '\t', ' '
        ]
        for seq in SEQUENCES_TO_REMOVE:
            answer = answer.replace(seq, '')
        for seq in SEQUENCES_TO_STOP:
            if seq in answer:
                answer = answer[:answer.index(seq)]        
        for char in CHARS_TO_DEDUPLICATE:
            answer = re.sub(f'{char}+', f'{char}', answer)
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
        inputs = QAEngine._preprocess_input(question, context)
        answer = self.llm_model._call(inputs)
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
