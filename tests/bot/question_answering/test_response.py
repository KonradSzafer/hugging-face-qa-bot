import pytest
from bot.question_answering.response import Response


def test_set_response():
    r = Response()
    r.set_response('Hello, World!')
    assert r.get_response() == 'Hello, World!'


def test_set_sources():
    r = Response()
    r.set_sources(['source1', 'source1', 'source2'])
    assert len(r.get_sources()) == 2


def test_get_sources_as_text():
    r = Response()
    r.set_sources(['source1', 'source2'])
    assert isinstance(r.get_sources_as_text(), str)


def test_get_response_include_sources():
    r = Response()
    r.set_response('Hello, World!')
    r.set_sources(['source1', 'source2'])
    assert len(r.get_response(include_sources=True)) > len('Hello, World!')
