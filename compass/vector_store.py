import os
from typing import Optional, List, Union
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from .compass import Compass

class VectorStore:
    """
    VectorStore merges file-level, class-level, and method-level data from Compass
    into cohesive Documents for final embedding, including usage-based structural info.
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
        if isinstance(source, Compass):
            self.vector_store = self._create_vectorstore()
            self.retriever = self._create_retriever()

    def _create_documents(self) -> List[Document]:
        if not isinstance(self.source, Compass):
            return []

        c = self.source
        all_docs = []

        # 1) File-level docs
        for fpath, data in c.file_summaries.items():
            summary = data.get("summary", "")
            doc = Document(
                page_content=summary,
                metadata={
                    "file_path": fpath,
                    "summary_level": "file"
                }
            )
            all_docs.append(doc)

        # 2) Class-level docs
        for cname, cdata in c.class_summaries.items():
            summary = cdata.get("summary","")
            doc = Document(
                page_content=summary,
                metadata={
                    "class_name": cname,
                    "summary_level": "class"
                }
            )
            all_docs.append(doc)

        # 3) Method-level docs
        for cname, methods_dict in c._class_methods_dict.items():
            for mname in methods_dict:
                full_symbol = f"{cname}::{mname}"
                msum = c.method_summaries.get(full_symbol, {})
                method_summary = msum.get("summary","")
                structural_fp = msum.get("structural_fingerprint","")
                code_text = msum.get("code","")

                merged_text = (
                    f"[Method Summary]\n{method_summary}\n\n"
                    f"[Structural Fingerprint]\n{structural_fp}\n\n"
                    f"[Raw Code]\n{code_text}"
                )
                doc = Document(
                    page_content=merged_text,
                    metadata={
                        "method_name": full_symbol,
                        "summary_level": "method"
                    }
                )
                all_docs.append(doc)

        # 4) Global methods
        for gm, gcode in c._global_methods_dict.items():
            msum = c.method_summaries.get(gm, {})
            method_summary = msum.get("summary","")
            structural_fp = msum.get("structural_fingerprint","")
            code_text = msum.get("code","")

            merged_text = (
                f"[Method Summary]\n{method_summary}\n\n"
                f"[Structural Fingerprint]\n{structural_fp}\n\n"
                f"[Raw Code]\n{code_text}"
            )
            doc = Document(
                page_content=merged_text,
                metadata={
                    "method_name": gm,
                    "summary_level": "method"
                }
            )
            all_docs.append(doc)

        return all_docs

    def _create_vectorstore(self) -> Chroma:
        from langchain_chroma import Chroma
        if self.persist_dir and os.path.exists(self.persist_dir):
            vs = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            if len(vs.get()['ids']) > 0:
                return vs

        docs = self._create_documents()
        vs = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        vs.add_documents(docs)
        return vs

    def _create_retriever(self):
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7}
        )

    @property
    def documents(self) -> List[Document]:
        coll = self.vector_store._collection
        result = coll.get(include=["documents"])
        docs = []
        for doc_obj in result["documents"]:
            if isinstance(doc_obj, dict):
                page_content = doc_obj.get("page_content","")
                meta = doc_obj.get("metadata",{})
            else:
                page_content = doc_obj
                meta = {}
            docs.append(Document(page_content=page_content, metadata=meta))
        return docs

    @property
    def embedding_matrix(self) -> np.ndarray:
        coll = self.vector_store._collection
        result = coll.get(include=['embeddings'])
        embs = result["embeddings"]
        arr = np.array(embs)
        return arr.reshape(-1, arr.shape[-1]) if arr.ndim == 2 else arr

    def get_documents_with_embeddings(self) -> tuple[List[Document], np.ndarray]:
        from langchain.schema import Document
        coll = self.vector_store._collection
        result = coll.get(include=["documents","embeddings","metadatas"])

        docs = []
        for d, meta in zip(result["documents"], result["metadatas"]):
            if isinstance(d, dict):
                page_content = d.get("page_content","")
                mm = d.get("metadata", {})
            else:
                page_content = d
                mm = {}
            if meta: 
                mm.update(meta)
            docs.append(Document(page_content=page_content, metadata=mm))
        embs = np.array(result["embeddings"])
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return docs, embs

    @classmethod
    def from_compass(cls, compass: Compass, persist: bool = False):
        return cls(source=compass, persist=persist)

    @classmethod
    def from_persist_storage(cls, persist_dir: str):
        instance = cls(persist=True)
        instance.persist_dir = persist_dir
        instance.vector_store = instance._create_vectorstore()
        instance.retriever = instance._create_retriever()
        return instance