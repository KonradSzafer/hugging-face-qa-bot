from typing import List


class Response:
    def __init__(self):
        self.answer = ''
        self.sources = []

    def set_answer(self, answer: str) -> None:
        self.answer = answer

    def set_sources(self, sources: List) -> None:
        self.sources = list(set([str(s) for s in sources]))

    def get_sources(self) -> List[str]:
        return self.sources

    def get_sources_as_text(self) -> str:
        sources_text = '\n\nSources:'
        for i, (source) in enumerate(self.sources):
            sources_text += f'\n [{i+1}] {source}'
        return sources_text

    def get_answer(self, include_sources: bool = False) -> str:
        answer = self.answer
        if include_sources:
            answer += self.get_sources_as_text()
        return answer

    def __str__(self):
        return self.get_answer(include_sources=True)
