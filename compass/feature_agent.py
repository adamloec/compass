from typing import List, Optional, Any, ClassVar
from pydantic import Field
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain

from .logger import Logger
LOGGER = Logger.create(__name__)

class FeatureAgent(Chain):
    """
    A chain that clusters documents into conceptual feature groups and generates
    a high-level feature/component name for each cluster using an LLM.

    Attributes:
        vector_store (Any): A vector store object with a `get_documents_with_embeddings()` method
            that returns documents and their corresponding embeddings. Expected to be compatible
            with the provided VectorStore interface.
        known_features (List[str]): A list of known high-level feature names. These serve as conceptual
            references or anchors to guide the naming of new features.
        model (ChatOpenAI): The LLM instance used to generate feature names. Default is a ChatOpenAI instance
            with model_name="gpt-4o-mini" and temperature=0 for deterministic responses.

    Class Attributes:
        input_keys (ClassVar[List[str]]): The input keys expected by this chain. This chain does not require inputs.
        output_keys (ClassVar[List[str]]): The output keys produced by this chain. Returns one key: "feature_dict".
        prompt_template (ClassVar[PromptTemplate]): A prompt template for generating feature names from cluster summaries.

    Methods:
        as_chain(vector_store, known_features, model): Class method to construct an instance of the chain easily.
        _get_docs_and_embeddings() -> tuple[List[Document], np.ndarray]: Retrieves documents and embeddings from the vector store.
        _determine_optimal_clusters(embedding_matrix: np.ndarray) -> int: Determines the optimal number of clusters based on coherence.
        _get_compass_features(docs: List[Document], embedding_matrix: np.ndarray) -> dict: Clusters docs into features and names each cluster.
        _call(inputs: dict) -> dict: Executes the entire pipeline and returns a dictionary with the "feature_dict" key.
    """

    vector_store: Any = Field(
        ...,
        description="The vector store object providing `get_documents_with_embeddings()` method."
    )
    known_features: List[str] = Field(
        default_factory=list,
        description="Known high-level feature names used as references for naming new features."
    )
    model: ChatOpenAI = Field(
        default_factory=lambda: ChatOpenAI(model_name="gpt-4o-mini", temperature=0), # Feature generation model will go here
        description="The LLM used to generate feature names for each cluster."
    )

    # The chain expects no input keys and produces a single output key: "feature_dict"
    input_keys: ClassVar[List[str]] = []
    output_keys: ClassVar[List[str]] = ["feature_dict"]

    prompt_template: ClassVar[PromptTemplate] = PromptTemplate(
        input_variables=["cluster_summaries", "reference_list_str"],
        template="""
            These function summaries belong to a single high-level feature or component group.

            We have a set of known features to guide the level of abstraction and style:
            {reference_list_str}

            These known features represent the kind of top-level, conceptual components we want: user-visible elements, 
            major system modules, or conceptual building blocks of the application. Think of them as anchors. 
            Your goal is to produce a feature name that fits naturally into a similar conceptual space, 
            not too low-level or purely technical. Consider these known features as pillars of the conceptual landscape. 
            If a cluster's purpose clearly aligns with one of the known features, you may refine or re-use that known feature name, 
            or invent a closely related conceptual feature that naturally complements or expands the existing set.

            Important instructions:
            - Provide exactly one concise, human-readable feature/component name.
            - The name should be at a similar conceptual level to the known features provided.
            - Do not provide multiple options or a list.
            - Do not introduce overly technical or micro-level concepts; maintain a high-level, user/system perspective.
            - If none of the known features fit perfectly, choose a new, similarly conceptual name that would make sense 
            to someone familiar with the known features.
            - Do not mention code or implementation details, only the conceptual feature.
            - Avoid camel case and underscores like "texture_manager" or "valueComparator".

            Summaries:
            {cluster_summaries}

            Now, provide a single, high-level feature/component name that aligns well with the known features and/or the provided summaries:
            """.strip(),
        )

    @classmethod
    def as_chain(cls, vector_store, known_features: Optional[List[str]] = None, model: Optional[ChatOpenAI] = None) -> "FeatureAgent":
        """
        Class method to create a FeatureAgent chain instance.

        Args:
            vector_store: The vector store object that provides documents and embeddings.
            known_features: Optional list of known feature names.
            num_features: Optional number of desired clusters.
            model: Optional LLM instance to use instead of the default ChatOpenAI model.

        Returns:
            A configured instance of FeatureAgent ready to be invoked.
        """
        LOGGER.info("Created feature generation chain.")

        return cls(
            vector_store=vector_store,
            known_features=known_features or [],
            model=model or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        )

    def _get_docs_and_embeddings(self) -> tuple[List[Document], np.ndarray]:
        """
        Retrieve documents and embeddings from the vector store.

        Returns:
            A tuple of (docs, embedding_matrix):
            - docs: A list of Document objects.
            - embedding_matrix: A NumPy array of embeddings corresponding to the docs.
        """
        LOGGER.debug("Getting docs and embeddings from vector storage.")

        docs, embedding_matrix = self.vector_store.get_documents_with_embeddings()
        return docs, embedding_matrix

    def _determine_optimal_clusters(self, embedding_matrix: np.ndarray) -> int:
        """
        Determine the optimal number of clusters.
        The method iterates over possible cluster counts and evaluates cluster coherence,
        penalizing small clusters. The cluster count with the highest score is chosen.

        Args:
            embedding_matrix (np.ndarray): The embedding matrix of shape (num_docs, embedding_dim).

        Returns:
            int: The chosen number of clusters.
        """
        LOGGER.debug("Determining optimal clusters for Agglomerative Clustering.")

        similarity_matrix = cosine_similarity(embedding_matrix)
        max_clusters = min(20, len(embedding_matrix) - 1)
        min_clusters = max(3, len(self.known_features))

        best_score = -1
        optimal_n = min_clusters

        # Try different cluster counts and pick the one with the best coherence score
        for n in range(min_clusters, max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n, metric='precomputed', linkage='complete')
            labels = clustering.fit_predict(1 - similarity_matrix)

            # Calculate coherence scores for each cluster
            coherence_scores = []
            for i in range(n):
                cluster_mask = labels == i

                if np.sum(cluster_mask) > 1:
                    cluster_sims = similarity_matrix[cluster_mask][:, cluster_mask]
                    coherence = np.mean(cluster_sims[np.triu_indices_from(cluster_sims, k=1)])
                    coherence_scores.append(coherence)

            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
            _, counts = np.unique(labels, return_counts=True)
            small_clusters = np.sum(counts < 3)
            size_penalty = 1 - (small_clusters / n) * 0.5
            score = avg_coherence * size_penalty

            if score > best_score:
                best_score = score
                optimal_n = n
        return optimal_n

    def _get_compass_features(self, docs: List[Document], embedding_matrix: np.ndarray) -> dict:
        """
        Cluster documents based on their embeddings, then generate a conceptual feature name 
        for each cluster using the LLM.

        Args:
            docs (List[Document]): The list of documents to cluster.
            embedding_matrix (np.ndarray): The embeddings for these documents.

        Returns:
            Dict[str, List[str]]: A dictionary mapping each generated feature name to a list of 
            associated method names (or doc identifiers).
        """
        LOGGER.debug("Getting all code base features based on compass summary clustering.")
        similarity_matrix = cosine_similarity(embedding_matrix)
        n_clusters = self._determine_optimal_clusters(embedding_matrix)
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
        
        distance_matrix = 1 - similarity_matrix
        labels = cluster_model.fit_predict(distance_matrix)

        summaries = [d.page_content for d in docs]
        method_names = [d.metadata.get("method_name", f"method_{i}") for i, d in enumerate(docs)]
        reference_list_str = "\n".join(self.known_features) if self.known_features else "None"
        feature_dict = {}

        # For each cluster, summarize and name the conceptual feature
        for cluster_id in range(n_clusters):
            cluster_summaries = [summaries[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not cluster_summaries:
                continue

            cluster_text = " ".join(cluster_summaries)
            prompt_val = self.prompt_template.format_prompt(
                cluster_summaries=cluster_text,
                reference_list_str=reference_list_str
            )
            response = self.model.invoke(prompt_val)

            # Extract the feature name from the LLM response
            feature_name = response.content.strip().split("\n")[0].strip(" -:*")

            methods = [method_names[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
            if feature_name not in feature_dict:
                feature_dict[feature_name] = []
            feature_dict[feature_name].extend(methods)

        return feature_dict

    def _call(self, inputs: dict) -> dict:
        """
        Execute the chain. Retrieves documents and embeddings, clusters them into conceptual features, 
        generates a feature name for each cluster, and returns the mapping.

        Args:
            inputs (dict): Ignored in this chain, as it does not require inputs.

        Returns:
            dict: A dictionary with a single key "feature_dict" mapping feature names to lists of method names.
        """
        LOGGER.info("Starting feature agent chain.")

        docs, embedding_matrix = self._get_docs_and_embeddings()
        feature_dict = self._get_compass_features(docs, embedding_matrix)
        return {"feature_dict": feature_dict}