from typing import List


class Response:
    def __init__(self):
        self.response = ''
        self.sources = []

    def set_response(self, response: str) -> None:
        self.response = response

    def set_sources(self, sources: List) -> None:
        self.sources = list(set([str(s) for s in sources]))

    def get_sources(self) -> List[str]:
        return self.sources

    def get_sources_as_text(self) -> str:
        sources_text = '\n\nSources:'
        for i, (source) in enumerate(self.sources):
            sources_text += f'\n [{i+1}] {source}'
        return sources_text

    def get_response(self, include_sources: bool = False) -> str:
        response = self.response
        if include_sources:
            response += self.get_sources_as_text()
        return response

    def __str__(self):
        return self.get_response(include_sources=True)
