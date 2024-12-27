from typing import List, Optional, Any, ClassVar
from pydantic import Field
import numpy as np
import itertools
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain

from ..logger import Logger
LOGGER = Logger.create(__name__)

class ForwardsFeatureAgent(Chain):
    """
    A robust multi-pass clustering agent that identifies and groups related code features using
    an iterative merge-split approach with adjacency weighting.

    Attributes:
        vector_store (Any): Store containing document embeddings and metadata with a 
            get_documents_with_embeddings() method.
        known_features (List[str]): List of existing feature names that serve as conceptual
            anchors to guide the clustering and naming process.
        model (ChatOpenAI): LLM model for making clustering decisions. Default is ChatOpenAI
            with model_name="gpt-4o-mini" and temperature=0.
        min_cluster_size (int): Minimum number of items allowed per cluster.

    Class Attributes:
        input_keys (ClassVar[List[str]]): Empty list as this chain requires no inputs.
        output_keys (ClassVar[List[str]]): Contains "feature_dict" as the only output key.
        cluster_name_prompt (ClassVar[PromptTemplate]): Template for generating feature names.
        merge_prompt (ClassVar[PromptTemplate]): Template for cluster merge decisions.
        split_prompt (ClassVar[PromptTemplate]): Template for cluster split decisions.

    Methods:
        as_chain(vector_store, known_features, model, min_cluster_size): Class method to construct chain instance.
        _get_docs_and_embeddings() -> tuple[List[Document], np.ndarray]: Gets docs and embeddings from store.
        _build_cluster_summary(docs, doc_indices, max_chars, max_per_doc) -> str: Creates cluster summaries.
        _build_adjacency_map(docs) -> dict: Maps relationships between documents.
        _adjacency_weighted_distance(base_dist, adjacency) -> np.ndarray: Weights distances by relationships.
        _initial_clustering(docs, embeddings) -> dict: Performs initial document clustering.
        _single_pass_merge_splits(cluster_dict, docs) -> dict: Executes one iteration of merges and splits.
        _attempt_merges(cluster_dict, docs) -> dict: Tries to merge similar clusters.
        _attempt_splits(cluster_dict, docs) -> dict: Tries to split heterogeneous clusters.
        _enforce_min_cluster_size(cluster_dict, docs) -> dict: Ensures minimum cluster sizes.
        _final_naming(cluster_dict, docs) -> dict: Assigns final feature names to clusters.
        _call(inputs) -> dict: Executes complete clustering pipeline.
    """

    vector_store: Any = Field(...)
    known_features: List[str] = Field(default_factory=list)
    model: ChatOpenAI = Field(
        default_factory=lambda: ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    )

    # Minimum cluster size
    min_cluster_size: int = 5

    input_keys: ClassVar[List[str]] = []
    output_keys: ClassVar[List[str]] = ["feature_dict"]

    cluster_name_prompt: ClassVar[PromptTemplate] = PromptTemplate(
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

    merge_prompt: ClassVar[PromptTemplate] = PromptTemplate(
        input_variables=["cluster_a_name", "cluster_a_summaries",
                         "cluster_b_name", "cluster_b_summaries"],
        template="""
            We have two feature clusters:

            Cluster A: "{cluster_a_name}"
            Summaries:
            {cluster_a_summaries}

            Cluster B: "{cluster_b_name}"
            Summaries:
            {cluster_b_summaries}

            Should these clusters be merged into a single conceptual feature?
            Respond with either "Yes" or "No" only.
            """.strip(),
    )

    split_prompt: ClassVar[PromptTemplate] = PromptTemplate(
        input_variables=["cluster_name", "cluster_summaries"],
        template="""
            Cluster: "{cluster_name}"
            Summaries:
            {cluster_summaries}

            Do these items actually form multiple distinct conceptual features?
            If so, how many sub-clusters? If it's cohesive, say "No split needed."

            Respond exactly in the format:
            "Split into X"
            or
            "No split needed"
            """.strip(),
    )

    @classmethod
    def as_chain(
        cls, 
        vector_store,
        known_features: Optional[List[str]] = None,
        model: Optional[ChatOpenAI] = None,
        min_cluster_size: int = 5
    ) -> "ForwardsFeatureAgent":
        return cls(
            vector_store=vector_store,
            known_features=known_features or [],
            model=model or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
            min_cluster_size=min_cluster_size
        )

    def _get_docs_and_embeddings(self) -> tuple[List[Document], np.ndarray]:
        """
        Retrieves all documents and their corresponding embeddings from the vector store. This method
        serves as the initial data gathering step for the clustering process, ensuring we have both
        the document content and their vector representations for similarity calculations.

        Args:
            None

        Returns:
            tuple: A pair containing:
                - List[Document]: List of document objects with content and metadata
                - np.ndarray: Matrix of document embeddings where each row corresponds to a document
        """
        return self.vector_store.get_documents_with_embeddings()

    def _build_cluster_summary(self, docs: List[Document], doc_indices: List[int], max_chars: int = 2000, max_per_doc: int = 500) -> str:
        """
        Creates a condensed summary of documents within a cluster by concatenating truncated versions
        of each document's content. This summary is used for LLM-based decision making about cluster
        operations. The method enforces both per-document and total character limits to prevent
        token limits from being exceeded in LLM calls.

        Args:
            docs: Complete list of all documents in the system
            doc_indices: List of indices identifying which documents belong to this cluster
            max_chars: Maximum total characters allowed in the complete summary (default: 2000)
            max_per_doc: Maximum characters to include from each individual document (default: 500)

        Returns:
            str: A concatenated string of document summaries, separated by newlines and truncated
                to respect both per-document and total length limits
        """
        out = []
        total = 0
        for idx in doc_indices:
            snippet = docs[idx].page_content[:max_per_doc]
            if total + len(snippet) > max_chars:
                break
            out.append(snippet)
            total += len(snippet)
        return "\n".join(out)

    def _build_adjacency_map(self, docs: List[Document]) -> dict:
        """
        Constructs a comprehensive map of relationships between code elements based on method calls
        and inheritance patterns. This creates a graph-like structure where edges represent different
        types of code relationships (calls, inherits, etc.) that will influence clustering decisions.

        The method analyzes metadata from each document to identify:
        - Method calls between different code elements
        - Inheritance relationships between classes
        - Bidirectional relationships (e.g., calls/called_by)

        Args:
            docs: List of document objects containing code elements and their metadata,
                 including information about method calls and inheritance

        Returns:
            dict: A nested dictionary where:
                - Outer key: Document index
                - Inner key: Related document index
                - Inner value: Relationship type (e.g., "calls", "inherits", "called_by", "inherited_by")
        """
        
        adjacency = {i: {} for i in range(len(docs))}
        
        # Build method name to index lookup
        method_to_idx = {}
        for i, doc in enumerate(docs):
            if doc.metadata.get("summary_level") == "method":
                method_name = doc.metadata.get("method_name")
                if method_name:
                    method_to_idx[method_name] = i
        
        # Process relationships
        for i, doc in enumerate(docs):
            if doc.metadata.get("summary_level") != "method":
                continue
            
            # Handle method calls
            if calls := doc.metadata.get("calls"):
                for called_method in calls.split(","):
                    if called_method and (called_idx := method_to_idx.get(called_method)):
                        adjacency[i][called_idx] = "calls"
                        adjacency[called_idx][i] = "called_by"  # Differentiate direction
            
            # Handle inheritance
            if inherits := doc.metadata.get("inherits_from"):
                for parent in inherits.split(","):
                    if parent and (parent_idx := method_to_idx.get(parent)):
                        adjacency[i][parent_idx] = "inherits"
                        adjacency[parent_idx][i] = "inherited_by"  # Differentiate direction
        
        return adjacency

    def _adjacency_weighted_distance(self, base_dist: np.ndarray, adjacency: dict) -> np.ndarray:
        """
        Modifies the base distance matrix by incorporating code relationship information to create
        a more semantically meaningful distance metric. Documents with direct relationships (like
        inheritance or method calls) will have their distances reduced according to relationship type.

        The method applies different weights based on relationship types:
        - Inheritance relationships receive strongest weight (0.4-0.5)
        - Method calls receive moderate weight (0.6-0.7)
        - Directionality of relationships affects weight strength

        Args:
            base_dist: Original distance matrix computed from embedding similarities
            adjacency: Map of document relationships from _build_adjacency_map

        Returns:
            np.ndarray: Modified distance matrix where distances between related documents
                       are reduced according to their relationship type and direction
        """
        dist_mat = base_dist.copy()
        
        # Define weights for different relationship types
        RELATIONSHIP_WEIGHTS = {
            "inherits": 0.4,      # Strongest connection - direct inheritance
            "inherited_by": 0.5,  # Strong connection - parent class
            "calls": 0.6,        # Moderate connection - direct method call
            "called_by": 0.7     # Weaker connection - being called by
        }
        
        for i in range(len(dist_mat)):
            for j, rel_type in adjacency[i].items():
                if i != j:
                    weight = RELATIONSHIP_WEIGHTS[rel_type]
                    dist_mat[i, j] *= weight
                    
        return dist_mat

    def _initial_clustering(self, docs: List[Document], embeddings: np.ndarray) -> dict:
        """
        Performs the initial clustering of documents using a combination of embedding similarities
        and code relationships. This creates the starting point for subsequent iterative refinement.

        The method:
        1. Computes cosine similarity between embeddings
        2. Converts similarities to distances
        3. Applies adjacency weighting to distances
        4. Uses hierarchical clustering to create initial groups
        5. Organizes results into a cluster dictionary

        Args:
            docs: List of all documents to be clustered
            embeddings: Matrix of document embeddings

        Returns:
            dict: Initial clustering results where:
                - Keys: Cluster identifiers (e.g., "Cluster_0")
                - Values: Lists of document indices belonging to each cluster
        """
        similarity = cosine_similarity(embeddings)
        distance = 1 - similarity
        adjacency = self._build_adjacency_map(docs)
        distance = self._adjacency_weighted_distance(distance, adjacency)

        n_clusters = min(8, len(docs))
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
        labels = model.fit_predict(distance)

        cluster_dict = {}
        for i, lbl in enumerate(labels):
            cluster_dict.setdefault(f"Cluster_{lbl}", []).append(i)
        return cluster_dict

    def _single_pass_merge_splits(self, cluster_dict: dict, docs: List[Document]) -> dict:
        """
        Executes a complete iteration of the cluster refinement process, including both merging
        similar clusters and splitting heterogeneous ones. This method maintains cluster quality
        through a sequence of operations:

        1. Attempts to merge similar clusters
        2. Enforces minimum cluster size through merging
        3. Identifies and splits heterogeneous clusters
        4. Re-enforces minimum cluster size after splits

        Args:
            cluster_dict: Current state of clustering, mapping cluster names to document indices
            docs: Complete list of documents for reference in decision-making

        Returns:
            dict: Updated clustering state after all merge and split operations
        """
        # merges
        cluster_dict = self._attempt_merges(cluster_dict, docs)
        # min size merges
        cluster_dict = self._enforce_min_cluster_size(cluster_dict, docs)
        # splits
        cluster_dict = self._attempt_splits(cluster_dict, docs)
        # min size merges again
        cluster_dict = self._enforce_min_cluster_size(cluster_dict, docs)
        return cluster_dict

    def _attempt_merges(self, cluster_dict: dict, docs: List[Document]) -> dict:
        """
        Systematically evaluates all possible cluster pairs for potential merging using LLM-based
        decision making. The method iteratively considers pairs of clusters and merges them if
        the LLM determines they represent the same conceptual feature.

        The process:
        1. Iterates through all cluster pairs
        2. For each pair, asks LLM if they should be merged
        3. If yes, combines clusters and updates the clustering state
        4. Continues until no more merges are possible or recommended

        Args:
            cluster_dict: Current clustering state mapping names to document indices
            docs: Complete list of documents for generating cluster summaries

        Returns:
            dict: Updated clustering state after all approved merges
        """
        cluster_names = list(cluster_dict.keys())
        i = 0
        while i < len(cluster_names):
            j = i + 1
            merged_any = False
            while j < len(cluster_names):
                cA = cluster_names[i]
                cB = cluster_names[j]
                if self._llm_says_merge(cA, cluster_dict[cA], cB, cluster_dict[cB], docs):
                    cluster_dict[cA].extend(cluster_dict[cB])
                    cluster_dict.pop(cB)
                    cluster_names.pop(j)
                    merged_any = True
                else:
                    j += 1
            if not merged_any:
                i += 1
        return cluster_dict

    def _llm_says_merge(self, cA_name: str, cA_docs: List[int], cB_name: str, cB_docs: List[int], docs: List[Document]) -> bool:
        """
        Consults the LLM to determine if two clusters should be merged based on their content
        and conceptual similarity. Provides the LLM with summaries of both clusters and expects
        a binary decision.

        The method:
        1. Generates summaries for both clusters
        2. Formats the merge decision prompt
        3. Interprets LLM's response as a yes/no decision

        Args:
            cA_name: Name of the first cluster
            cA_docs: Document indices in first cluster
            cB_name: Name of the second cluster
            cB_docs: Document indices in second cluster
            docs: Complete list of documents for generating summaries

        Returns:
            bool: True if LLM recommends merging the clusters, False otherwise
        """
        # build truncated summaries
        a_summaries = self._build_cluster_summary(docs, cA_docs)
        b_summaries = self._build_cluster_summary(docs, cB_docs)
        prompt = self.merge_prompt.format_prompt(
            cluster_a_name=cA_name,
            cluster_a_summaries=a_summaries,
            cluster_b_name=cB_name,
            cluster_b_summaries=b_summaries
        )
        resp = self.model.invoke(prompt).content.strip()
        return resp.lower().startswith("yes")

    def _attempt_splits(self, cluster_dict: dict, docs: List[Document]) -> dict:
        """
        Examines each cluster for potential subdivision into more cohesive subclusters using
        LLM guidance. For clusters identified as containing multiple distinct concepts, performs
        hierarchical clustering to create appropriate subclusters.

        The process:
        1. Evaluates each cluster for potential splitting
        2. If split is recommended, determines optimal number of subclusters
        3. Performs hierarchical clustering on the subset
        4. Creates new clusters from the split results

        Args:
            cluster_dict: Current clustering state
            docs: Complete list of documents for analysis

        Returns:
            dict: Updated clustering state after all approved splits
        """
        import copy
        new_dict = copy.deepcopy(cluster_dict)

        for cname in list(new_dict.keys()):
            doc_indices = new_dict[cname]
            # build truncated summary
            cluster_summaries = self._build_cluster_summary(docs, doc_indices)
            split_into = self._llm_says_split(cname, cluster_summaries)
            if split_into and split_into > 1:
                sub_labels = self._subcluster(doc_indices, docs, split_into)
                for sub_id in range(split_into):
                    sub_name = f"{cname}_sub{sub_id+1}"
                    new_dict[sub_name] = [doc_indices[i] for i, lbl in enumerate(sub_labels) if lbl == sub_id]
                new_dict.pop(cname, None)
        return new_dict

    def _llm_says_split(self, cname: str, cluster_summaries: str) -> Optional[int]:
        """
        Asks LLM if a cluster should be split and into how many parts.

        Args:
            cname: Name of cluster
            cluster_summaries: Text summaries of items in cluster

        Returns:
            Optional[int]: Number of subclusters to split into, or None if no split needed
        """
        prompt = self.split_prompt.format_prompt(
            cluster_name=cname,
            cluster_summaries=cluster_summaries
        )
        resp = self.model.invoke(prompt).content.strip().lower()
        if resp.startswith("split into"):
            parts = resp.split()
            if len(parts) >= 3:
                try:
                    return int(parts[2])
                except:
                    return None
        return None

    def _subcluster(self, doc_indices: List[int], docs: List[Document], k: int) -> np.ndarray:
        """
        Performs subclustering on a set of documents.

        Args:
            doc_indices: Indices of documents to subcluster
            docs: List of all documents
            k: Number of subclusters to create

        Returns:
            np.ndarray: Array of cluster labels
        """
        _, full_embeddings = self._get_docs_and_embeddings()
        sub_embs = full_embeddings[doc_indices]
        dist = 1 - cosine_similarity(sub_embs)
        model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        labels = model.fit_predict(dist)
        return labels#

    def _enforce_min_cluster_size(self, cluster_dict: dict, docs: List[Document]) -> dict:
        """
        Ensures all clusters meet the minimum size requirement by merging small clusters into
        their most similar larger neighbors. Uses embedding centroids to determine the best
        merge targets for small clusters.

        The process:
        1. Computes centroid embeddings for all clusters
        2. Identifies clusters below minimum size
        3. Finds most similar larger cluster based on centroid similarity
        4. Merges small clusters into their best matches
        5. Updates centroids after merges

        Args:
            cluster_dict: Current clustering state
            docs: Complete list of documents

        Returns:
            dict: Updated clustering state with all clusters meeting minimum size requirement
        """
        from copy import deepcopy
        cluster_dict = deepcopy(cluster_dict)

        _, full_embeddings = self._get_docs_and_embeddings()

        def centroid(indices: List[int]) -> np.ndarray:
            embs = full_embeddings[indices]
            return np.mean(embs, axis=0)

        cluster_names = list(cluster_dict.keys())
        cluster_centroids = {cname: centroid(cluster_dict[cname]) for cname in cluster_names}

        for cname in cluster_names:
            if cname not in cluster_dict:
                continue
            if len(cluster_dict[cname]) < self.min_cluster_size:
                best_target = None
                best_sim = -999
                c1 = cluster_centroids[cname]
                for other in cluster_names:
                    if other == cname or other not in cluster_dict:
                        continue
                    c2 = cluster_centroids[other]
                    sim = cosine_similarity(c1.reshape(1, -1), c2.reshape(1, -1))[0, 0]
                    if sim > best_sim:
                        best_sim = sim
                        best_target = other
                if best_target is not None:
                    cluster_dict[best_target].extend(cluster_dict[cname])
                    cluster_dict.pop(cname)
                    new_embs = full_embeddings[cluster_dict[best_target]]
                    cluster_centroids[best_target] = np.mean(new_embs, axis=0)

        return cluster_dict

    def _final_naming(self, cluster_dict: dict, docs: List[Document]) -> dict:
        """
        Generates final, human-readable feature names for each cluster using LLM guidance and
        known feature names as reference points. Ensures unique names by adding numbers to
        duplicates.

        The process:
        1. For each cluster, generates a summary
        2. Provides summary and known features to LLM
        3. Processes LLM response into a clean feature name
        4. Handles naming conflicts by adding numbers
        5. Maps final names to document lists

        Args:
            cluster_dict: Final clustering state before naming
            docs: Complete list of documents

        Returns:
            dict: Mapping of human-readable feature names to lists of document identifiers
        """
        final_dict = {}
        reference_list_str = "\n".join(self.known_features) if self.known_features else "None"

        for cname, dindices in cluster_dict.items():
            # again, build truncated summary for naming
            cluster_summaries = self._build_cluster_summary(docs, dindices)
            prompt_val = self.cluster_name_prompt.format_prompt(
                cluster_summaries=cluster_summaries,
                reference_list_str=reference_list_str
            )
            response = self.model.invoke(prompt_val)
            feature_name = response.content.strip().split("\n")[0].strip(" -:*")

            if feature_name in final_dict:
                final_dict[feature_name].extend(dindices)
            else:
                final_dict[feature_name] = dindices

        return final_dict

    def _call(self, inputs: dict) -> dict:
        """
        Executes the complete feature identification pipeline, from initial clustering through
        iterative refinement to final naming. This is the main entry point for the clustering
        process.

        The pipeline:
        1. Retrieves documents and embeddings
        2. Performs initial clustering with adjacency weighting
        3. Iteratively refines clusters through merges and splits
        4. Assigns final feature names
        5. Converts results to the expected output format

        Args:
            inputs: Dictionary of input parameters (unused in current implementation)

        Returns:
            dict: Contains single key "feature_dict" mapping feature names to lists of
                 code elements belonging to each feature
        """
        docs, embeddings = self._get_docs_and_embeddings()

        # 1) initial adjacency-based cluster
        cluster_dict = self._initial_clustering(docs, embeddings)

        # 2) iterative merges + splits
        max_iterations = 5
        for it in range(max_iterations):
            LOGGER.debug(f"Iteration {it+1}: {len(cluster_dict)} clusters")
            old_count = len(cluster_dict)
            new_dict = self._single_pass_merge_splits(cluster_dict, docs)
            new_count = len(new_dict)

            LOGGER.debug(f"Iteration {it+1}: {old_count} -> {new_count} clusters")
            if new_dict.keys() == cluster_dict.keys():
                stable = True
                for k in new_dict:
                    if set(new_dict[k]) != set(cluster_dict.get(k, [])):
                        stable = False
                        break
                if stable:
                    cluster_dict = new_dict
                    break
                else:
                    cluster_dict = new_dict
            else:
                cluster_dict = new_dict

        # 3) final naming
        final_clusters = self._final_naming(cluster_dict, docs)

        # 4) build final feature_dict with doc identifiers
        feature_dict = {}
        for feature_name, dindices in final_clusters.items():
            items = []
            for idx in dindices:
                m = docs[idx].metadata.get("method_name")
                if m:
                    items.append(m)
                else:
                    fpath = docs[idx].metadata.get("file_path")
                    items.append(fpath if fpath else f"doc_{idx}")
            feature_dict[feature_name] = items

        return {"feature_dict": feature_dict}