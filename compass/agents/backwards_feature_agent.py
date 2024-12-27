from typing import List, Optional, Any, ClassVar
from pydantic import Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain

from ..logger import Logger
LOGGER = Logger.create(__name__)

class BackwardsFeatureAgent(Chain):
    """
    A robust multi-step chain that identifies features through a top-down approach:

    1. Retrieves and processes all document summaries from Compass
    2. Dynamically chunks summaries to respect token limits
    3. Recursively merges partial summaries into a complete codebase understanding
    4. Uses LLM to propose high-level features from the unified summary
    5. Assigns individual code elements to the most relevant features

    Key advantages:
        - Token-aware processing prevents context window overflows
        - Hierarchical summary building maintains context coherence
        - Top-down approach ensures consistent feature granularity
        - Embedding-based assignment ensures accurate code-to-feature mapping

    Attributes:
        vector_store (Any): Store containing document embeddings and metadata
        model (ChatOpenAI): LLM for generating summaries and features
        known_features (List[str]): Reference features for maintaining consistency
        max_tokens_prompt (int): Maximum tokens allowed in a single prompt

    Class Attributes:
        input_keys (ClassVar[List[str]]): Empty list as this chain requires no inputs
        output_keys (ClassVar[List[str]]): Contains "feature_dict" as the only output key
        partial_merge_prompt (ClassVar[PromptTemplate]): Template for merging partial summaries
        propose_features_prompt (ClassVar[PromptTemplate]): Template for feature generation
        refine_feature_prompt (ClassVar[PromptTemplate]): Template for feature name refinement
        merge_features_prompt (ClassVar[PromptTemplate]): Template for merging features
    """

    vector_store: Any = Field(...)
    model: ChatOpenAI = Field(
        default_factory=lambda: ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    )
    known_features: List[str] = Field(default_factory=list)
    max_tokens_prompt: int = 7000
    min_cluster_size: int = Field(default=5)

    input_keys: ClassVar[List[str]] = []
    output_keys: ClassVar[List[str]] = ["feature_dict"]

    ###########################################################################
    # PROMPTS
    ###########################################################################

    partial_merge_prompt: ClassVar[PromptTemplate] = PromptTemplate(
        input_variables=["combined_text"],
        template="""
            The following are multiple code or method summaries from a software project:
            {combined_text}

            Please merge them into a single cohesive summary, 
            focusing on high-level functionality or major components. 
            Avoid repeating details verbatim.
            """.strip()
    )

    propose_features_prompt: ClassVar[PromptTemplate] = PromptTemplate(
        input_variables=["big_summary"],
        template="""
            Based on this high-level codebase summary, identify the major features or components present:

            {big_summary}

            Important instructions:
            - List only top-level, conceptual features that represent:
              * User-visible elements
              * Major system modules
              * Conceptual building blocks of the application
            - Provide each feature as a bullet point with a concise, human-readable name
            - Focus on high-level, user/system perspective features
            - Do not include technical implementation details or low-level concepts
            - Avoid camel case and underscores like "texture_manager" or "valueComparator"
            - No additional commentary, just one feature name per line

            List the features:
            """.strip(),
    )

    refine_feature_prompt: ClassVar[PromptTemplate] = PromptTemplate(
        input_variables=["raw_feature_name"],
        template="""
            The bullet item was: {raw_feature_name}

            Convert this into a single concise, conceptual feature name:
            """.strip()
    )

    merge_features_prompt: ClassVar[PromptTemplate] = PromptTemplate(
        input_variables=["feature_a", "feature_b", "summaries_a", "summaries_b"],
        template="""
            We have two features:

            Feature A: "{feature_a}"
            Summaries:
            {summaries_a}

            Feature B: "{feature_b}"
            Summaries:
            {summaries_b}

            Are these describing essentially the same conceptual feature? 
            Respond with "Yes" or "No" only.
            """.strip(),
    )

    ###########################################################################
    # Implementation
    ###########################################################################

    @classmethod
    def as_chain(
        cls,
        vector_store,
        model: Optional[ChatOpenAI] = None,
        known_features: Optional[List[str]] = None,
        max_tokens_prompt: int = 10000,
        min_cluster_size: int = 5
    ) -> "BackwardsFeatureAgent":
        """
        Creates a configured instance of the BackwardsFeatureAgent chain.

        Args:
            vector_store: Vector store containing document embeddings
            model: Optional LLM model override
            known_features: Optional list of reference feature names
            max_tokens_prompt: Maximum tokens per prompt
            min_cluster_size: Minimum cluster size for merging

        Returns:
            BackwardsFeatureAgent: Configured chain instance
        """
        LOGGER.info("Creating backwards feature generation chain.")
        return cls(
            vector_store=vector_store,
            model=model or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
            known_features=known_features or [],
            max_tokens_prompt=max_tokens_prompt,
            min_cluster_size=min_cluster_size
        )

    def _get_docs_and_embeddings(self) -> tuple[List[Document], np.ndarray]:
        """
        Retrieves documents and their embeddings from the vector store.

        Returns:
            tuple: (documents, embeddings) pair
        """
        LOGGER.debug("Retrieving documents and embeddings from vector store.")
        return self.vector_store.get_documents_with_embeddings()

    def _recursive_summarize_docs(self, docs: List[Document]) -> str:
        """
        Recursively summarizes documents into a single coherent summary while respecting
        token limits. Uses dynamic chunking and hierarchical merging to maintain context.

        Args:
            docs: List of documents to summarize

        Returns:
            str: Unified summary of all documents
        """
        LOGGER.debug(f"Starting recursive summarization of {len(docs)} documents.")
        if not docs:
            return "No documents to summarize."

        combined_text = self._join_doc_summaries(docs)
        if self._approx_token_count(combined_text) < self.max_tokens_prompt:
            return self._llm_merge(combined_text)

        subgroups = self._split_docs_by_tokens(docs, max_tokens=self.max_tokens_prompt // 2)
        LOGGER.debug(f"Split documents into {len(subgroups)} subgroups for processing.")
        
        partial_summaries = []
        for i, group in enumerate(subgroups):
            LOGGER.debug(f"Processing subgroup {i+1}/{len(subgroups)}")
            text_block = self._join_doc_summaries(group)
            summary = self._llm_merge(text_block)
            partial_summaries.append(summary)

        partial_docs = [Document(page_content=s) for s in partial_summaries]
        return self._recursive_summarize_docs(partial_docs)

    def _split_docs_by_tokens(self, docs: List[Document], max_tokens: int) -> List[List[Document]]:
        """
        Partitions documents into groups that fit within token limits.

        Args:
            docs: Documents to partition
            max_tokens: Maximum tokens per group

        Returns:
            List[List[Document]]: List of document groups
        """
        LOGGER.debug(f"Splitting {len(docs)} documents into token-sized chunks.")
        groups = []
        current_group = []
        current_text_tokens = 0

        for d in docs:
            d_text = self._doc_as_text(d)
            doc_tokens = self._approx_token_count(d_text)
            if current_text_tokens + doc_tokens > max_tokens and current_group:
                groups.append(current_group)
                current_group = [d]
                current_text_tokens = doc_tokens
            else:
                current_group.append(d)
                current_text_tokens += doc_tokens

        if current_group:
            groups.append(current_group)

        LOGGER.debug(f"Created {len(groups)} document groups.")
        return groups

    def _join_doc_summaries(self, docs: List[Document]) -> str:
        """
        Join doc-level summaries into a single big text block.
        We'll label each doc so there's some context.
        """
        lines = []
        for d in docs:
            lines.append(self._doc_as_text(d))
        return "\n\n".join(lines)

    def _doc_as_text(self, d: Document) -> str:
        label = d.metadata.get("method_name") or d.metadata.get("file_path") or "doc"
        return f"[{label}]\n{d.page_content}"

    def _llm_merge(self, combined_text: str) -> str:
        """
        Actually prompt the LLM with partial_merge_prompt to unify summaries.
        """
        prompt_val = self.partial_merge_prompt.format_prompt(combined_text=combined_text)
        resp = self.model.invoke(prompt_val)
        return resp.content.strip()

    ###########################################################################
    # Approx Token Counting
    ###########################################################################
    def _approx_token_count(self, text: str) -> int:
        """
        A naive approach: assume average 4 characters per token.
        If you have tiktoken installed, do a real token count:
          e.g.:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            tokens = enc.encode(text)
            return len(tokens)
        """
        return len(text) // 4

    ###########################################################################
    # Step 2: Propose Features
    ###########################################################################
    def _propose_features(self, big_summary: str) -> List[str]:
        prompt_val = self.propose_features_prompt.format_prompt(big_summary=big_summary)
        resp = self.model.invoke(prompt_val).content.strip()
        lines = resp.split("\n")
        raw_features = [ln.strip("-* ") for ln in lines if ln.strip()]
        return raw_features

    def _refine_feature_list(self, raw_features: List[str]) -> List[str]:
        """
        If bullet items are verbose, quickly refine them.
        Also include known_features if desired.
        """
        refined = []
        for rf in raw_features:
            prompt_val = self.refine_feature_prompt.format_prompt(raw_feature_name=rf)
            rresp = self.model.invoke(prompt_val).content.strip()
            rresp = rresp.strip("-:*")
            if rresp:
                refined.append(rresp)

        # Optionally unify with known_features
        final_features = self.known_features + refined
        return final_features

    ###########################################################################
    # Step 3: Assign Docs to Each Feature
    ###########################################################################
    def _assign_docs_to_feature(
        self,
        docs: List[Document],
        embeddings: np.ndarray,
        feature_name: str,
        top_k: int = 10,
        min_sim_threshold: float = 0.3
    ) -> List[int]:
        """
        For each feature, embed the name & measure similarity 
        to each doc embedding. Return top_k docs above a threshold.
        """
        feature_vec = self.vector_store.embeddings.embed_query(feature_name)
        results = []
        for i, doc_vec in enumerate(embeddings):
            score = cosine_similarity(
                doc_vec.reshape(1, -1),
                np.array(feature_vec).reshape(1, -1)
            )[0,0]
            results.append((i, score))

        results.sort(key=lambda x: x[1], reverse=True)
        assigned = []
        for idx, sc in results[:top_k]:
            if sc >= min_sim_threshold:
                assigned.append(idx)
        return assigned

    def _build_cluster_summaries(self, doc_indices: List[int], docs: List[Document]) -> str:
        lines = []
        for idx in doc_indices[:10]:  # limit to 10 for brevity
            snippet = docs[idx].page_content[:200]
            lines.append(snippet)
        return "\n".join(lines)

    def _llm_says_merge(self, fA: str, fB: str, summA: str, summB: str) -> bool:
        prompt_val = self.merge_features_prompt.format_prompt(
            feature_a=fA,
            feature_b=fB,
            summaries_a=summA,
            summaries_b=summB
        )
        resp = self.model.invoke(prompt_val).content.strip().lower()
        return resp.startswith("yes")

    def _merge_small_or_duplicate_features(
        self, 
        feature_map: dict, 
        docs: List[Document], 
        embeddings: np.ndarray
    ) -> dict:
        LOGGER.debug("Starting feature merge process.")
        feature_map = self._attempt_llm_feature_merges(feature_map, docs)
        feature_map = self._merge_tiny_clusters(feature_map, embeddings)
        return feature_map

    def _attempt_llm_feature_merges(self, feature_map: dict, docs: List[Document]) -> dict:
        LOGGER.debug("Attempting LLM-based feature merges.")
        feature_names = list(feature_map.keys())
        i = 0
        while i < len(feature_names):
            j = i + 1
            merged_any = False
            while j < len(feature_names):
                fA = feature_names[i]
                fB = feature_names[j]
                if fA not in feature_map or fB not in feature_map:
                    j += 1
                    continue

                summA = self._build_cluster_summaries(feature_map[fA], docs)
                summB = self._build_cluster_summaries(feature_map[fB], docs)

                if self._llm_says_merge(fA, fB, summA, summB):
                    LOGGER.debug(f"Merging features '{fA}' and '{fB}'")
                    feature_map[fA].extend(feature_map[fB])
                    feature_map.pop(fB)
                    feature_names.pop(j)
                    merged_any = True
                else:
                    j += 1

            if not merged_any:
                i += 1

        return feature_map

    def _merge_tiny_clusters(self, feature_map: dict, embeddings: np.ndarray) -> dict:
        LOGGER.debug(f"Merging clusters smaller than {self.min_cluster_size}")
        from copy import deepcopy
        updated_map = deepcopy(feature_map)

        centroids = {}
        for fname, idxs in updated_map.items():
            if not idxs:
                centroids[fname] = None
            else:
                doc_vecs = [embeddings[i] for i in idxs]
                centroids[fname] = np.mean(doc_vecs, axis=0)

        feature_names = list(updated_map.keys())

        for fname in feature_names:
            if fname not in updated_map:
                continue
            cluster_size = len(updated_map[fname])
            if cluster_size < self.min_cluster_size and cluster_size > 0:
                LOGGER.debug(f"Processing small cluster '{fname}' (size: {cluster_size})")
                best_target = None
                best_sim = -999
                c1 = centroids[fname]
                if c1 is None:
                    updated_map.pop(fname)
                    continue
                for other in feature_names:
                    if other == fname or other not in updated_map:
                        continue
                    c2 = centroids[other]
                    if c2 is None:
                        continue
                    sim_val = cosine_similarity(
                        c1.reshape(1,-1),
                        c2.reshape(1,-1)
                    )[0,0]
                    if sim_val > best_sim:
                        best_sim = sim_val
                        best_target = other

                if best_target is not None:
                    LOGGER.debug(f"Merging '{fname}' into '{best_target}'")
                    updated_map[best_target].extend(updated_map[fname])
                    updated_map.pop(fname)
                    doc_vecs = [embeddings[i] for i in updated_map[best_target]]
                    centroids[best_target] = np.mean(doc_vecs, axis=0)

        return updated_map

    def _call(self, inputs: dict) -> dict:
        """
        Executes the complete feature identification pipeline:
        1. Retrieves and processes documents
        2. Builds hierarchical summary
        3. Generates feature proposals
        4. Assigns documents to features

        Args:
            inputs: Unused in this implementation

        Returns:
            dict: Contains "feature_dict" mapping features to code elements
        """
        LOGGER.info("Starting backwards feature identification process.")
        
        docs, embeddings = self._get_docs_and_embeddings()
        LOGGER.info(f"Retrieved {len(docs)} documents from vector store.")

        big_summary = self._recursive_summarize_docs(docs)
        LOGGER.info("Generated unified codebase summary.")

        raw_features = self._propose_features(big_summary)
        feature_names = self._refine_feature_list(raw_features)
        LOGGER.info(f"Generated {len(feature_names)} feature proposals.")

        # Create initial feature assignments
        initial_map = {}
        for fname in feature_names:
            assigned_indices = self._assign_docs_to_feature(docs, embeddings, fname)
            initial_map[fname] = assigned_indices

        # Merge similar or small features
        final_map = self._merge_small_or_duplicate_features(initial_map, docs, embeddings)

        # Convert to final format
        feature_dict = {}
        for fname, indices in final_map.items():
            doc_labels = []
            for idx in indices:
                m = docs[idx].metadata.get("method_name")
                if m:
                    doc_labels.append(m)
                else:
                    fpath = docs[idx].metadata.get("file_path")
                    doc_labels.append(fpath if fpath else f"doc_{idx}")
            feature_dict[fname] = doc_labels
            LOGGER.debug(f"Assigned {len(doc_labels)} elements to feature '{fname}'")

        LOGGER.info("Completed feature identification process.")
        return {"feature_dict": feature_dict}