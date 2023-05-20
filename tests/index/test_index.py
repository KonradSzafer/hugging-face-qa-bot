import pytest
from typing import Any
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

@pytest.fixture(scope="module")
def embedding_model() -> HuggingFaceInstructEmbeddings:
    model_name = "hkunlp/instructor-large"
    embed_instruction = "Represent the Hugging Face library documentation"
    query_instruction = "Query the most relevant piece of information from the Hugging Face documentation"
    return HuggingFaceInstructEmbeddings(
        model_name=model_name,
        embed_instruction=embed_instruction,
        query_instruction=query_instruction,
    )

@pytest.fixture(scope="module")
def index_path() -> str:
    return "index/"

@pytest.fixture(scope="module")
def index(embedding_model: HuggingFaceInstructEmbeddings, index_path: str):
    return FAISS.load_local(index_path, embedding_model)

@pytest.fixture(scope="module")
def query() -> str:
    return "How to use the tokenizer?"

def test_load_index(embedding_model: HuggingFaceInstructEmbeddings, index_path: str):
    index = FAISS.load_local(index_path, embedding_model)
    assert index is not None, "Failed to load index"

def test_index_page_content(index, query: str):
    query_docs = index.similarity_search(query=query, k=3)
    assert isinstance(query_docs[0].page_content, str)

def test_index_metadata(index, query):
    query_docs = index.similarity_search(query=query, k=3)
    assert isinstance(query_docs[0].metadata['source'], str)
