# vector_store.py

import os
from enum import Enum
from typing import Optional, List, Union
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .compass import Compass

class VectorStore:
    """
    VectorStore merges file-level and method-level summaries from Compass,
    plus newly discovered inheritance info, into documents.
    """

    def __init__(self, source: Optional[Union[Compass, str]] = None, persist: bool = False):
        self.source = source
        self.persist = persist
        self.persist_dir = None
        
        if self.source and self.persist:
            if isinstance(source, Compass):
                repo_name = os.path.basename(str(source.dir_path)).lower()
            else:
                repo_name = os.path.basename(source).lower()
            self.persist_dir = f"chroma_db/{repo_name}_vdb"
            
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = None
        self.retriever = None

    @property
    def embedding_matrix(self) -> np.ndarray:
        collection = self.vector_store._collection
        result = collection.get(include=['embeddings'])
        embeddings = result["embeddings"]
        if not embeddings:
            raise ValueError("No embeddings found in vector store")
        matrix = np.array(embeddings)
        return matrix.reshape(-1, matrix.shape[-1]) if matrix.ndim == 1 else matrix

    @property
    def documents(self) -> List[Document]:
        collection = self.vector_store._collection
        result = collection.get(include=['documents'])
        return [
            Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {})) 
            for doc in result["documents"]
        ]

    @classmethod
    def from_persist_storage(cls, persist_dir: str):
        instance = cls(persist=True)
        instance.persist_dir = persist_dir
        instance.vector_store = instance._create_vectorstore()
        instance.retriever = instance._create_retriever()
        return instance
    
    @classmethod
    def from_compass(cls, compass: Compass, persist: bool = False):
        instance = cls(source=compass, persist=persist)
        instance.vector_store = instance._create_vectorstore()
        instance.retriever = instance._create_retriever()
        return instance
    
    def get_documents_with_embeddings(self) -> tuple[List[Document], np.ndarray]:
        collection = self.vector_store._collection
        result = collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        docs = [
            Document(page_content=doc, metadata=metadata or {})
            for doc, metadata in zip(result["documents"], result["metadatas"])
        ]
        embeddings = np.array(result["embeddings"])
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return docs, embeddings

    def _create_documents(self) -> List[Document]:
        """
        If source is Compass, combine file-level + method-level + inheritance info
        into Documents. If it's a raw directory, just chunk code files.
        """
        if isinstance(self.source, Compass):
            return self._create_compass_documents()
        return self._create_directory_documents()

    def _create_compass_documents(self) -> List[Document]:
        compass_source = self.source
        all_documents = []

        # 1) File-level
        for file_name, file_data in compass_source.file_summaries.items():
            if "summary" in file_data:
                all_documents.append(
                    Document(
                        page_content=file_data["summary"],
                        metadata={
                            "file_path": file_name,
                            "file_code": file_data.get("code", ""),
                            "summary_level": "file"
                        }
                    )
                )

        # 2) Method-level
        for method_name, method_data in compass_source.method_summaries.items():
            if "summary" in method_data:
                # retrieve calls (set) from compass
                calls_list = list(compass_source._method_call_dict.get(method_name, []))
                doc = Document(
                    page_content=method_data["summary"],
                    metadata={
                        "method_name": method_name,
                        "calls": ",".join(calls_list),  # store as comma-separated string
                        "code": method_data.get("code", ""),
                        "summary_level": "method"
                    }
                )
                # Add file_path to method-level documents
                if "file_path" in method_data:
                    doc.metadata["file_path"] = method_data["file_path"]
                all_documents.append(doc)

        # 3) Classes and summaries
        for class_name, class_data in compass_source.class_summaries.items():
            if "summary" in class_data:
                class_methods = list(class_data.get("methods", {}).keys())
                
                all_documents.append(
                    Document(
                        page_content=class_data["summary"],
                        metadata={
                            "class_name": class_name,
                            "methods": ",".join(class_methods),
                            "summary_level": "class"
                        }
                    )
                )

        return all_documents
    
    def _create_vectorstore(self) -> Chroma:
        try:
            if self.persist_dir and os.path.exists(self.persist_dir):
                vector_store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                if len(vector_store.get()['ids']) > 0:
                    return vector_store
            
            if self.source is None:
                raise ValueError(f"No source provided for {self.__class__.__name__}")
            
            documents = self._create_documents()
            for doc in documents:
                if "calls" in doc.metadata and isinstance(doc.metadata["calls"], str):
                    pass
                if "inherits_from" in doc.metadata and not isinstance(doc.metadata["inherits_from"], str):
                    doc.metadata["inherits_from"] = ",".join(doc.metadata["inherits_from"])

            vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            vector_store.add_documents(documents)
            return vector_store
            
        except Exception as e:
            print(f"Failed to create vector store {self.__class__.__name__}: {str(e)}")

    def _create_retriever(self):
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7}
        )
        return retriever