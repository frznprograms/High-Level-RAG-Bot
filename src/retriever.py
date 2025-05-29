from vector_store import VectorStore
from data_loader import Dataloader

class Retriever:
    def __init__(self, k=2):
        self.k = k
        self.vector_store = VectorStore()
        self.retriever = None

    def initialize(self, chunks):
        self.vector_store.load_or_create_store(chunks=chunks)
        self.retriever = self.vector_store.store.as_retriever(search_kwargs={"k": self.k})

    def retrieve_relevant_data(self, query):
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
        return self.retriever.invoke(query)
    

if __name__ == "__main__":
    loader = Dataloader()
    documents = loader.load_documents()
    splits = loader.chunk_documents()

    retriever = Retriever()
    retriever.initialize(splits)
    results = retriever.retrieve_relevant_data(
        query="When was Star Wars first released in cinemas?"
    )
    # print(results)