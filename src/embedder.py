import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from data_loader import Dataloader

load_dotenv()

class Embedder:
    def __init__(self):
        self.embeddings = None

    def embed_as_vectors(self, chunks):
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            openai_api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        )
        chunk_embeddings = embeddings.embed_documents([
            chunk.page_content for chunk in chunks
        ])
        self.embeddings = chunk_embeddings
        print(f"Created embeddings for {len(chunk_embeddings)} chunks.")
        return chunk_embeddings

if __name__ == "__main__":
    loader = Dataloader()
    documents = loader.load_documents()
    chunks = loader.chunk_documents()
    embedder = Embedder()
    chunk_embeddings = embedder.embed_as_vectors(chunks)
    # first 5 entries of first embedding
    print(chunk_embeddings[0][:5])