import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from llama_index import GPTFaissIndex, SimpleDirectoryReader, LangchainEmbedding


def create_index():
    model_name = "hkunlp/instructor-large"
    embed_instruction = "Represent the Hugging Face library documentation"
    query_instruction = "Query the most relevant piece of information from the Hugging Face documentation"

    embedding = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        embed_instruction=embed_instruction,
        query_instruction=query_instruction
    )
    embedding = LangchainEmbedding(embedding)

    d = 768
    faiss_index = faiss.IndexFlatL2(d)
    documents = SimpleDirectoryReader('./datasets/hf/').load_data()
    index = GPTFaissIndex(documents, embed_model=embedding, faiss_index=faiss_index)
    index.save_to_disk(
        'index_faiss.json',
        faiss_index_save_path="index_faiss_core.index"
    )

    # example query
    print(index.query("How to install transformers library?"))


if __name__ == '__main__':
    create_index()
