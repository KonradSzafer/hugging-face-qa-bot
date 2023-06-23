from typing import Mapping, Optional, List, Any
import os
from langchain.llms.base import LLM

class MockLocalBinaryModel(LLM):
    """
    Mock Local Binary Model class, used for generating the string "a".

    Args:
        model_id (str): The ID of the model to be mocked.

    Attributes:
        model_path (str): The path to the model to be mocked.
        llm (str): The string "a".

    Raises:
        ValueError: If the model_path does not exist.
    """

    model_path: str = None
    llm: str = {"mock": "READY TO MOCK"}

    def __init__(self, model_id: str = None):
        super().__init__()
        self.model_path = f'bot/question_answering/{model_id}'
        if not os.path.exists(self.model_path):
            raise ValueError(f'{self.model_path} does not exist')


    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.llm

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_path}

    @property
    def _llm_type(self) -> str:
        return self.model_path
