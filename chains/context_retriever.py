import os, json
from typing import Optional, Dict, Any, List
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

SUPPORTED_DATA_TYPES = ["compass", "repo"]
SUPPORTED_RETRIEVAL_TYPES = ["similarity"]

class ContextRetriever:
    
    def __init__(self, data_path: str, data_type: str="compass", retrieval_type: str="similarity"):
        if data_type not in SUPPORTED_DATA_TYPES:
            raise Exception(f"Please use a supported data type: {SUPPORTED_DATA_TYPES}")
        if retrieval_type not in SUPPORTED_RETRIEVAL_TYPES:
            raise Exception(f"Please use a supported retrieval type: {SUPPORTED_RETRIEVAL_TYPES}")
    
        self.data_path = data_path
        self.data_type = data_type
        self.retrieval_type = retrieval_type
        
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        self.retriever = None
        self._initialize_retriever()

    # Loads compass json file into documents, file
    def _create_compass_documents(self) -> list[Document]:

        try:
            all_documents = []
            with open(self.data_path, 'r') as f:
                data = json.load(f)

            for file_path_key, functions in data.items():
                if isinstance(functions, dict):
                    for func_name, summary in functions.items():
                        # Split long summaries into chunks
                        chunks = self.text_splitter.split_text(summary)
                        for chunk in chunks:
                            all_documents.append({
                                "page_content": chunk,
                                "metadata": {
                                    "file_path": file_path_key,
                                    "function_name": func_name
                                }
                            })
            
            all_documents = [Document(page_context=doc["page_content"], metadata=doc["metadata"]) for doc in all_documents]
            return all_documents
        
        except Exception as e:
            print(f"Error processing {self.data_path}: {str(e)}")

    # Load entire code base into documents for embeddings
    def _create_repo_documents(self) -> list[Document]:
        
        try:
            all_documents = []
            documents = []
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith((".py", ".js", ".cpp", ".c", ".h")):
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            content = f.read()
                            documents.append(Document(page_content=content, metadata={"source": file}))

            for doc in documents:
                split_docs = self.text_splitter.split_documents([doc])
                all_documents.extend(split_docs)

            return all_documents

        except Exception as e:
            print(f"Error processing {self.data_path}: {str(e)}")

    def _create_vectorstore(self) -> Chroma:

        try:
            documents = None
            if self.data_type == "compass":
                documents = self._create_compass_documents()
            if self.data_type == "repo":
                print("Generating file documents")
                documents = self._create_repo_documents()

            vector_store = Chroma(
                embedding_function=self.embeddings
            )
            vector_store.add_documents(documents)

            return vector_store
        
        except Exception as e:
            print(f"Failed to create vector store {self.__class__.__name__}: {str(e)}")

    def _initialize_retriever(self) -> None:

        vector_store = self._create_vectorstore()
        
        if self.retrieval_type == "similarity":
            self.retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})