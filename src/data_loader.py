import json 
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Dataloader: 
    def __init__(self):
        self.documents = None
        self.chunks = None

    def load_documents(self, relative_folder_name="data"):
        # Get the absolute path to the current file (assumes this file is in src/)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Go one directory up and then into the data folder
        folder_path = os.path.join(base_dir, "..", relative_folder_name)
        folder_path = os.path.abspath(folder_path)
        documents = []
        counter = 0
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not file_path.endswith(".pdf"):
                print(f"Skipping file of unsupported type: {filename}")
                continue
            if file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
            else: 
                raise TypeError(f"Unsupported file type: {filename}")
            documents.extend(loader.load())
            counter += 1
        print(f"Total of {counter} documents loaded successfully.")
        self.documents = documents
        return documents
    
    def chunk_documents(
            self, 
            splitter=RecursiveCharacterTextSplitter,
            chunk_size=1000,
            chunk_overlap=200,
            len_func=len
    ):
        text_splitter = splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len_func
        )       
        splits = text_splitter.split_documents(self.documents)
        self.splits = splits
        print(f"Total of {len(splits)} chunks obtained from corpus.")
        return splits

if __name__ == "__main__":
    loader = Dataloader()
    documents = loader.load_documents()
    splits = loader.chunk_documents()
    print("First split: ")
    print(loader.splits[1])
    print("First split metadata: ")
    print(splits[1].metadata)

