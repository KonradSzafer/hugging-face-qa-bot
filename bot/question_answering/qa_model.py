from langchain import PromptTemplate, HuggingFaceHub, LLMChain


class LangChainModel:
    def __init__(
        self,
        hf_api_key: str,
        model_id: str,
    ):
        self.model = HuggingFaceHub(
            repo_id=model_id,
            model_kwargs={
                'max_length': 500,
                'length_penalty': 2,
                'num_beams': 16,
                'no_repeat_ngram_size': 2,
                'temperature': 0.8,
                'top_k': 150,
                'top_p': 0.9
            }, 
            huggingfacehub_api_token=hf_api_key
        )
        self.template = 'Question: {context}\n{question}'
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=['context', 'question']
        )
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.model)


    def get_answer(self, context: str, question: str) -> str:
        return self.llm_chain.run(
            context=context,
            question=question
        )
