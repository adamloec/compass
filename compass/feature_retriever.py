import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .compass import Compass
from .compass import Logger

LOGGER = Logger.create(__name__)

class CompassFeatureRetriever:

    def __init__(self, compass: Compass, known_features=None, num_features: Optional[int] = None):

        self.method_summaries = compass.method_summaries
        self.known_features = known_features if known_features else []
        self.num_features = num_features

        self.model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()

    @classmethod
    def get_features(cls, compass: Compass, known_features: List[str] = None, num_features: Optional[int] = None) -> Dict[str, List[str]]:

        LOGGER.info("Starting feature extraction")
        instance = cls(compass, known_features, num_features)
        docs, embedding_matrix = instance._create_compass_clusters()
        feature_dict = instance._get_compass_features(docs, embedding_matrix)
        
        LOGGER.info("Feature extraction completed")
        return feature_dict

    def _create_compass_clusters(self) -> Tuple[List[Document], np.ndarray]:

        LOGGER.info("Creating document embeddings and clusters.")
        
        summaries = []
        fn_keys = []
        for fn_key, data in self.method_summaries.items():
            if 'summary' in data:
                summaries.append(data['summary'])
                fn_keys.append(fn_key)

        docs = [Document(page_content=summ, metadata={"function_key": fn_key}) 
               for summ, fn_key in zip(summaries, fn_keys)]
        doc_texts = [d.page_content for d in docs]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)
        
        embedding_matrix = np.array(doc_embeddings)
        return docs, embedding_matrix

    def _determine_optimal_clusters(self, embedding_matrix: np.ndarray) -> int:

        if self.num_features:
            return self.num_features

        similarity_matrix = cosine_similarity(embedding_matrix)
        
        max_clusters = min(20, len(embedding_matrix) - 1)
        min_clusters = max(3, len(self.known_features))
        
        best_score = -1
        optimal_n = min_clusters
        
        for n in range(min_clusters, max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n, metric='precomputed', linkage='complete')
            labels = clustering.fit_predict(1 - similarity_matrix)
            
            coherence_scores = []
            for i in range(n):
                cluster_mask = labels == i
                if np.sum(cluster_mask) > 1:
                    cluster_similarities = similarity_matrix[cluster_mask][:, cluster_mask]
                    coherence = np.mean(cluster_similarities[np.triu_indices_from(cluster_similarities, k=1)])
                    coherence_scores.append(coherence)
            
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
            
            unique, counts = np.unique(labels, return_counts=True)
            small_clusters = np.sum(counts < 3)
            size_penalty = 1 - (small_clusters / n) * 0.5
            
            score = avg_coherence * size_penalty
            
            if score > best_score:
                best_score = score
                optimal_n = n
        
        return optimal_n

    def _get_compass_features(self, docs: List[Document], embedding_matrix: np.ndarray) -> Dict[str, List[str]]:

        LOGGER.info("Getting feature names from compass clusters.")
        
        similarity_matrix = cosine_similarity(embedding_matrix)
        distance_matrix = 1 - similarity_matrix
        
        n_clusters = self._determine_optimal_clusters(embedding_matrix)
        LOGGER.info(f"Using {n_clusters} clusters based on optimization")
        
        cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='complete'
        )
        labels = cluster_model.fit_predict(distance_matrix)

        summaries = [d.page_content for d in docs]
        fn_keys = [d.metadata["function_key"] for d in docs]
        
        feature_dict = defaultdict(list)
        
        for cluster_id in range(n_clusters):

            cluster_summaries = [summaries[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not cluster_summaries:
                continue
                
            reference_list_str = "\n".join(self.known_features) if self.known_features else "None"
            
            prompt = f"""
            These function summaries belong to a single high-level feature or component group.

            We have a set of known features to guide the level of abstraction and style:
            {reference_list_str}

            These known features represent the kind of top-level, conceptual components we want: user-visible elements, 
            major system modules, or conceptual building blocks of the application. Think of them as anchors. 
            Your goal is to produce a feature name that fits naturally into a similar conceptual space, 
            not too low-level or purely technical.

            Important instructions:
            - Provide exactly one concise, human-readable feature/component name.
            - The name should be at a similar conceptual level to the known features provided.
            - Do not provide multiple options or a list.
            - Do not introduce overly technical or micro-level concepts; maintain a high-level, user/system perspective.
            - If none of the known features fit perfectly, choose a new, similarly conceptual name.
            - Do not mention code or implementation details, only the conceptual feature.

            Summaries to analyze:
            {' '.join(cluster_summaries)}

            Now, provide a single, high-level feature/component name that aligns well with the known features and/or the provided summaries:
            """
            
            resp = self.model.invoke(prompt).content.strip()
            resp_lines = resp.split("\n")
            feature_name = resp_lines[0].strip(" -:*")
            
            LOGGER.debug(f"Cluster {cluster_id} named: {feature_name}")
            
            funcs = [fn_keys[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
            feature_dict[feature_name].extend(funcs)
            
        return dict(feature_dict)