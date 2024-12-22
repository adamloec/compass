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

from .logger import Logger
LOGGER = Logger.create(__name__)

class IterativeFeatureAgent(Chain):
    """
    A robust multi-pass merges+splits approach with adjacency weighting, 
    min-cluster-size merges, and text truncation to avoid overloading the LLM 
    with huge cluster summaries.
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

    ############################################################################
    # PROMPTS
    ############################################################################

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

    ############################################################################
    # MAIN METHODS
    ############################################################################

    @classmethod
    def as_chain(
        cls, 
        vector_store,
        known_features: Optional[List[str]] = None,
        model: Optional[ChatOpenAI] = None,
        min_cluster_size: int = 5
    ) -> "IterativeFeatureAgent":
        return cls(
            vector_store=vector_store,
            known_features=known_features or [],
            model=model or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
            min_cluster_size=min_cluster_size
        )

    def _get_docs_and_embeddings(self) -> tuple[List[Document], np.ndarray]:
        return self.vector_store.get_documents_with_embeddings()

    ############################################################################
    # TEXT CHUNKING / TRUNCATION
    ############################################################################

    def _build_cluster_summary(
        self,
        docs: List[Document],
        doc_indices: List[int],
        max_chars: int = 2000,
        max_per_doc: int = 500
    ) -> str:
        """
        Collect partial summaries from each doc, but do not exceed `max_chars` total 
        or `max_per_doc` for each doc summary. This prevents enormous inputs.
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

    ############################################################################
    # ADJACENCY EXAMPLE (INHERITANCE or CALLS) - (Optional)
    ############################################################################

    def _build_adjacency_map(self, docs: List[Document]) -> dict:
        """
        If you have inheritance or calls, unify them here. 
        For brevity, we show a no-op adjacency map.
        """
        adjacency = {i: set() for i in range(len(docs))}
        return adjacency

    def _adjacency_weighted_distance(self, base_dist: np.ndarray, adjacency: dict) -> np.ndarray:
        dist_mat = base_dist.copy()
        # optional weighting
        for i in range(len(dist_mat)):
            for j in adjacency[i]:
                if i != j:
                    dist_mat[i, j] *= 0.7
                    dist_mat[j, i] *= 0.7
        return dist_mat

    ############################################################################
    # INITIAL CLUSTER
    ############################################################################

    def _initial_clustering(self, docs: List[Document], embeddings: np.ndarray) -> dict:
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

    ############################################################################
    # LLM MERGES + SPLITS
    ############################################################################

    def _single_pass_merge_splits(self, cluster_dict: dict, docs: List[Document]) -> dict:
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
        _, full_embeddings = self._get_docs_and_embeddings()
        sub_embs = full_embeddings[doc_indices]
        dist = 1 - cosine_similarity(sub_embs)
        model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        labels = model.fit_predict(dist)
        return labels

    ############################################################################
    # MIN CLUSTER SIZE
    ############################################################################

    def _enforce_min_cluster_size(self, cluster_dict: dict, docs: List[Document]) -> dict:
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

    ############################################################################
    # FINAL NAMING
    ############################################################################

    def _final_naming(self, cluster_dict: dict, docs: List[Document]) -> dict:
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

            base_name = feature_name
            ctr = 2
            while feature_name in final_dict:
                feature_name = f"{base_name} ({ctr})"
                ctr += 1

            final_dict[feature_name] = dindices

        return final_dict

    ############################################################################
    # MAIN CALL
    ############################################################################

    def _call(self, inputs: dict) -> dict:
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