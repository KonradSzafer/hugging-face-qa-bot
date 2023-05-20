import pytest
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


def get_embedding_model():
    model_name = "hkunlp/instructor-large"
    embed_instruction = "Represent the Hugging Face library documentation"
    query_instruction = "Query the most relevant piece of information from the Hugging Face documentation"
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        embed_instruction=embed_instruction,
        query_instruction=query_instruction,
    )
    return embedding_model


INDEX_PATH = "index/"
EMBEDDING_MODEL = get_embedding_model()


def test_load_index():
    index = FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL)
    assert index is not None


def test_index_page_content():
    index = FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL)
    docs = index.similarity_search(
        query="How to use the tokenizer?", k=3)
    assert isinstance(docs[0].page_content, str)


def test_index_metadata():
    index = FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL)
    docs = index.similarity_search(
        query="How to use the tokenizer?", k=3)
    assert isinstance(docs[0].metadata['source'], str)
