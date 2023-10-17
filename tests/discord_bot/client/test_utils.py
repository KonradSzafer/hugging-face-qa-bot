import pytest
import os
from discord_bot.client.utils import ( \
    find_max_split_index, \
    find_max_split_index_from_sequence, \
    split_text_into_chunks
)


@pytest.fixture(scope='module')
def test_chunk() -> str:
    return 't. , \n .'


@pytest.fixture(scope='module')
def test_text() -> str:
    with open('tests/discord_bot/client/lorem_ipsum.txt', 'r') as f:
        text = f.read()
    assert text is not None, 'test text is empty'
    return text


def test_find_max_splitting_index(test_chunk: str):
    index = find_max_split_index(test_chunk, char='\n')
    assert index == 6, 'index should be 6'
    index = find_max_split_index(test_chunk, char='. ')
    assert index == 3, 'index should be 3'
    index = find_max_split_index(test_chunk, char='.')
    assert index == 8, 'index should be 8'


def test_find_max_split_index_from_sequence(test_chunk: str):
    index = find_max_split_index_from_sequence(
        test_chunk,
        split_characters=['\n']
    )
    assert index == 6, 'index should be 6'
    index = find_max_split_index_from_sequence(
        test_chunk,
        split_characters=['.', ', ', '\n']
    )
    assert index == 8, 'index should be 8'


def test_split_text_into_chunks_with_split_characters(test_text: str):
    max_chunk_size = 250
    chunks = split_text_into_chunks(
        test_text,
        split_characters=['. ', ', ', '\n'],
        min_size=20,
        max_size=max_chunk_size
    )
    for chunk in chunks:
        assert len(chunk) > 0, 'Chunk length is zero'
        assert len(chunk) <= max_chunk_size, 'Chunk length exceeds maximum limit'


def test_split_text_into_chunks_without_split_characters():
    test_text = 'a' * 1000
    max_chunk_size = 250
    chunks = split_text_into_chunks(
        test_text,
        split_characters=[],
        min_size=20,
        max_size=max_chunk_size
    )
    for chunk in chunks:
        assert len(chunk) == max_chunk_size, \
            'Chunk length is too small'
