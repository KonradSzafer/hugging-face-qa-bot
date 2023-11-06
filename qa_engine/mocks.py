import os
from typing import Mapping, Optional, Any

from langchain.llms.base import LLM


class MockLocalBinaryModel(LLM):
    """
    Mock Local Binary Model class.
    """

    model_path: str = None
    llm: str = 'Mocked Response'

    def __init__(self):
        super().__init__()

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        return self.llm

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'name_of_model': 'mock'}

    @property
    def _llm_type(self) -> str:
        return 'mock'
