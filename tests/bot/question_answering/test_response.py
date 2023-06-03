import pytest
from bot.question_answering.response import Response


def test_set_response():
    r = Response()
    r.set_response('Hello, World!')
    assert r.get_response() == 'Hello, World!'


def test_set_sources():
    r = Response()
    r.set_sources(['source1', 'source1', 'source2'])
    assert r.get_sources() == ['source1', 'source2']


def test_get_sources_as_text():
    r = Response()
    r.set_sources(['source1', 'source2'])
    assert r.get_sources_as_text() == \
        '\n\nSources:\n [1] source1\n [2] source2'


def test_get_response_include_sources():
    r = Response()
    r.set_response('Hello, World!')
    r.set_sources(['source1', 'source2'])
    assert r.get_response(include_sources=True) == \
        'Hello, World!\n\nSources:\n [1] source1\n [2] source2'


def test_str_method():
    r = Response()
    r.set_response('Hello, World!')
    r.set_sources(['source1', 'source2'])
    assert str(r) == 'Hello, World!\n\nSources:\n [1] source1\n [2] source2'
