import os
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma

class FeatureExtractorPipeline:
    def __init__(self, known_features=None):

        self.known_features = known_features if known_features else []

        self.file_methods_path = "file_methods_dict.json"
        self.method_call_path = "method_call_dict.json"
        self.method_code_path = "method_code_dict.json"
        
        with open(self.file_methods_path, "r") as f:
            self.file_methods_dict = json.load(f)
        with open(self.method_call_path, "r") as f:
            self.method_call_dict = json.load(f)
        with open(self.method_code_path, "r") as f:
            self.original_method_code_dict = json.load(f)

        self.chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.enriched_method_code_dict = {}

    def build_summaries(self):
        print("Building summaries for methods...")
        for filename, methods in self.file_methods_dict.items():
            for method_name, method_code in methods.items():
                if method_name in self.original_method_code_dict:
                    existing_code = self.original_method_code_dict[method_name]
                    if isinstance(existing_code, str):
                        self.enriched_method_code_dict[method_name] = {"code": existing_code}
                    else:
                        self.enriched_method_code_dict[method_name] = existing_code
                else:
                    self.enriched_method_code_dict[method_name] = {"code": method_code}

                if "summary" not in self.enriched_method_code_dict[method_name]:
                    print(f"Generating summary for method: {method_name}")
                    prompt = f"""
                            You are a highly skilled software engineer. Given the following code, provide a short, high-level summary of what this function or method does in simple terms. Focus on the underlying feature or component it relates to (like a UI element, a piece in a game, or a major system component), not just the code itself.

                            Code:
                            {self.enriched_method_code_dict[method_name]['code']}
                            Summary:
                            """
                    summary = self.chat_model.predict(prompt).strip()
                    self.enriched_method_code_dict[method_name]["summary"] = summary

        print("Summaries generated. Saving to file...")
        with open("compass_summaries.json", "w") as f:
            json.dump(self.enriched_method_code_dict, f, indent=2)
        print("Summaries saved successfully.")

    def create_vector_store(self):
        print("Creating vector store from summaries...")
        summaries = []
        fn_keys = []
        for fn_key, data in self.enriched_method_code_dict.items():
            if 'summary' in data:
                summaries.append(data['summary'])
                fn_keys.append(fn_key)

        print("Generating embeddings...")
        docs = [Document(page_content=summ, metadata={"function_key": fn_key}) for summ, fn_key in zip(summaries, fn_keys)]
        doc_texts = [d.page_content for d in docs]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)

        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)

        print("Indexing documents into Chroma vector store...")
        vectorstore = Chroma(collection_name="summaries",
                             persist_directory=self.persist_directory,
                             embedding_function=self.embeddings)
        vectorstore.add_texts(texts=doc_texts, metadatas=[d.metadata for d in docs], ids=fn_keys, embeddings=doc_embeddings)
        vectorstore.persist()
        print("Documents indexed successfully.")

        embedding_matrix = np.array(doc_embeddings)
        return docs, embedding_matrix

    def cluster_features(self, docs, embedding_matrix):
        print("Clustering features into high-level components...")
        cluster_model = AgglomerativeClustering(n_clusters=10)
        labels = cluster_model.fit_predict(embedding_matrix)

        summaries = [d.page_content for d in docs]
        feature_names = []
        for cluster_id in range(10):
            cluster_summaries = [summaries[i] for i, lbl in enumerate(labels) if lbl == cluster_id]

            reference_list_str = "\n".join(self.known_features) if self.known_features else "None"
            print(f"Naming cluster {cluster_id} with {len(cluster_summaries)} methods...")

            prompt = f"""
                    These function summaries belong to a single high-level feature or component group. 
                    We have a known set of reference features we consider relevant (e.g.):
                    {reference_list_str}

                    Think of these summaries in terms of what a user or a major subsystem would conceptually recognize.

                    Do not focus on code terms. Focus on conceptual features or components a user might know. 
                    If something aligns with one of the known features, reuse that name. 
                    If it's different, propose a new feature that fits well alongside them.

                    Summaries:
                    {cluster_summaries}

                    Suggest a concise, human-readable feature/component name:
                    """
            resp = self.chat_model.predict(prompt).strip()
            feature_name = resp if resp else f"Feature_{cluster_id}"
            feature_names.append(feature_name)
            print(f"Cluster {cluster_id} named: {feature_name}")

        fn_keys = [d.metadata["function_key"] for d in docs]
        feature_dict = {}
        for cluster_id, feature_name in enumerate(feature_names):
            funcs = [fn_keys[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
            feature_dict[feature_name] = funcs

        with open('feature_clusters.json', 'w') as f:
            json.dump(feature_dict, f, indent=2)
        print("Feature clusters saved.")

        return feature_dict

    def run_pipeline(self):
        print("Running pipeline...")
        self.build_summaries()
        docs, embedding_matrix = self.create_vector_store()
        feature_dict = self.cluster_features(docs, embedding_matrix)
        print("Pipeline completed.")
        return feature_dict