import os
from enum import Enum
from typing import Optional, List, Union
from dataclasses import dataclass
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .compass import Compass

class DataSourceType(Enum):
    """Enumeration of the different data source types for the vector store."""
    COMPASS = "compass"
    DIRECTORY = "directory"

class VectorStore:
    """
    A VectorStore object that constructs a vector database of documents from either
    Compass-generated summaries or a directory of code files. It provides methods to
    retrieve documents and their embeddings, facilitating similarity search and retrieval.

    Attributes:
        source (Union[Compass, str]): The data source, either a Compass object or a directory path.
        source_type (DataSourceType): Indicates whether the source is Compass or a directory.
        text_splitter (RecursiveCharacterTextSplitter): Used to chunk text into manageable pieces for embedding.
        embeddings (OpenAIEmbeddings): The embeddings model used to transform text into vector representations.
        vector_store (Chroma): The Chroma-based vector store that holds documents and their embeddings.
        retriever: A retriever interface for similarity search on the vector store.

    Methods:
        as_retriever(source, embedding_model_path): Class method that returns a retriever from a newly created vector store.
        get_documents_with_embeddings() -> (List[Document], np.ndarray): Retrieves all documents with their corresponding embeddings.
        documents: Property that returns all documents in the vector store as `Document` objects.
        embedding_matrix: Property that returns a NumPy array of all embeddings in the vector store.
    """
    def __init__(self, source: Union[Compass, str], embedding_model_path: Optional[str] = None):
        """
        Vector Store object that creates a vector database from either Compass data or a directory of code files.

        Args:
            source: Either Compass object, or a directory path.
            embedding_model_path: Optional path to embedding model. If None, uses default path
        """
        self.source = source
        self.source_type = (
            DataSourceType.COMPASS 
            if isinstance(source, Compass) 
            else DataSourceType.DIRECTORY
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        self.embeddings = OpenAIEmbeddings() # Embedding model will go here
        self.vector_store = self._create_vectorstore()
        self.retriever = self._create_retriever()

    @property
    def embedding_matrix(self) -> np.ndarray:
        """
        Retrieve the embedding matrix of all documents in the vector store.

        Returns:
            np.ndarray: A NumPy array of shape (num_docs, embedding_dim) containing all embeddings.

        Raises:
            ValueError: If no embeddings are found in the vector store.
        """
        collection = self.vector_store._collection
        result = collection.get(include=['embeddings'])
        embeddings = result["embeddings"]
        
        if not embeddings:
            raise ValueError("No embeddings found in vector store")
            
        matrix = np.array(embeddings)
        return matrix.reshape(-1, matrix.shape[-1]) if matrix.ndim == 1 else matrix

    @property
    def documents(self) -> List[Document]:
        """
        Retrieve all documents from the vector store as `Document` objects.

        Returns:
            List[Document]: A list of Document objects with `page_content` and optionally `metadata`.
        """
        collection = self.vector_store._collection
        result = collection.get(include=['documents'])
        return [Document(**doc) for doc in result["documents"]]

    @classmethod
    def as_retriever(cls, source: Union[Compass, str], embedding_model_path: Optional[str] = None):
        """
        Create a VectorStore instance from the given source and return its retriever.

        Args:
            source (Union[Compass, str]): The data source (Compass or directory).
            embedding_model_path (Optional[str]): Path to a custom embedding model (optional).

        Returns:
            A retriever object that can be used for similarity-based document retrieval.
        """
        instance = cls(source, embedding_model_path)
        return instance.retriever
    
    def get_documents_with_embeddings(self) -> tuple[List[Document], np.ndarray]:
        """
        Retrieve all documents and their embeddings from the vector store.

        Returns:
            (List[Document], np.ndarray): A tuple containing:
                - A list of `Document` objects.
                - A 2D NumPy array of embeddings corresponding to these documents.

        Raises:
            ValueError: If embeddings are not found or empty.
        """
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
        Create documents from the source.

        Returns:
            List[Document]: A list of Document objects extracted from either Compass or a directory.
        """
        if self.source_type == DataSourceType.COMPASS:
            return self._create_compass_documents()
        return self._create_directory_documents()

    def _create_compass_documents(self) -> List[Document]:
        """
        If the source is a Compass object, create documents from its method summaries.

        Returns:
            List[Document]: A list of Document objects derived from Compass method summaries.
        """
        compass_source = self.source
        all_documents = []
        
        for method_name, method_data in compass_source.method_summaries.items():
            if "summary" in method_data:
                all_documents.append(Document(
                    page_content=method_data.get("summary"),
                    metadata={
                        "method_name": method_name,
                        "code": method_data.get("code", "")
                    }
                ))
        
        return all_documents

    def _create_directory_documents(self) -> List[Document]:
        """
        If the source is a directory, read code files and split them into chunks for embedding.

        Returns:
            List[Document]: A list of Document objects created by reading and chunking code files.
        """
        dir_source = self.source
        all_documents = []
        
        for root, _, files in os.walk(dir_source.dir_path):
            for file in files:
                if file.endswith(".cpp", ".c", ".h", ".hpp", ".js", ".ts", ".tsx", ".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        chunks = self.text_splitter.split_text(content)
                        for chunk in chunks:
                            all_documents.append(Document(
                                page_content=chunk,
                                metadata={
                                    "file_path": file_path,
                                    "file_name": file
                                }
                            ))
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
                        continue
        
        return all_documents

    def _create_vectorstore(self) -> Chroma:
        """
        Create and populate the Chroma vector store with documents and their embeddings.

        Returns:
            Chroma: The populated Chroma vector store.

        Raises:
            Exception: If vector store creation fails.
        """
        try:
            documents = self._create_documents()
            vector_store = Chroma(embedding_function=self.embeddings)
            vector_store.add_documents(documents)

            return vector_store
        
        except Exception as e:
            print(f"Failed to create vector store {self.__class__.__name__}: {str(e)}")

    def _create_retriever(self):
        """
        Create a retriever for similarity-based document search.

        Returns:
            A retriever object configured with a similarity score threshold for filtering results.
        """
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7}
        )

        return retriever