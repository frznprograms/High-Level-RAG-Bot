import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from data_loader import Dataloader

load_dotenv()

class VectorStore:
    def __init__(self):
        self.persist_directory = "./chroma_db"
        self.collection_name = "my_collection"
        self.embedding_function = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            openai_api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        )
        self.store = None

    def store_exists(self):
        return os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3"))

    def load_or_create_store(self, chunks=None):
        if self.store_exists():
            print("Existing vector store found. Loading from disk...")
            self.store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
        elif chunks is not None:
            print("No existing store found. Creating new vector store...")
            self.store = Chroma.from_documents(
                collection_name=self.collection_name,
                documents=chunks,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
            print("Vector store created and persisted to disk.")
        else:
            raise ValueError("No existing store found and no chunks provided to create one.")

    def get_matches(self, query, k=2):
        if self.store is None: 
            raise ValueError("Vector store has not been created. Call load_or_create_store() first.")

        search_results = self.store.similarity_search(query, k=k)
        print(f"Top {k} mosty relevant chunks for the query: \n")
        for i, result in enumerate(search_results, 1):
            print(f"Result {i}:")
            print(f"Source: {result.metadata.get('source', 'Unknown')}")
            # print(f"Content: {result.page_content}")
            print()

if __name__ == "__main__":
    loader = Dataloader()
    documents = loader.load_documents()
    splits = loader.chunk_documents()
    vectorstore = VectorStore()
    vectorstore.load_or_create_store(chunks=splits)
    # vectorstore.load_or_create_store()
    vectorstore.get_matches(
        query="When was Star Wars first released in cinemas?"
    )