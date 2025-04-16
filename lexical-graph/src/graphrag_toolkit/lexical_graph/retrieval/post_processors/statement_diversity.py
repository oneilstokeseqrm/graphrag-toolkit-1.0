# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import re
import spacy
from typing import List, Optional, Any, Callable
from pydantic import Field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from graphrag_toolkit.lexical_graph import ModelError
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResult

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode

logger = logging.getLogger(__name__)

def _all_text(node:BaseNode) -> str:
    return node.text

def _topics_and_statements(node:BaseNode) -> str:
    lines = []
    search_result = SearchResult.model_validate_json(node.text)
    lines.append(search_result.topic)
    for statement in search_result.statements:
        lines.append(statement)
    return '\n'.join(lines)

def _topics(node:BaseNode) -> str:
    lines = []
    search_result = SearchResult.model_validate_json(node.text)
    return search_result.topic

ALL_TEXT = _all_text
TOPICS_AND_STATEMENTS = _topics_and_statements
TOPICS = _topics

class StatementDiversityPostProcessor(BaseNodePostprocessor):
    """Removes similar statements using TF-IDF similarity."""
    
    similarity_threshold: float = Field(default=0.975)
    nlp: Any = Field(default=None)
    text_fn: Callable[[BaseNode], str] = Field(default=None)

    def __init__(self, similarity_threshold: float = 0.975, text_fn = None):
        super().__init__(
            similarity_threshold=similarity_threshold,
            text_fn = text_fn or ALL_TEXT
        )
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
            self.nlp.add_pipe('sentencizer')
        except OSError:
            raise ModelError("Please install the spaCy model using: python -m spacy download en_core_web_sm")

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts using optimized spaCy configuration."""
        preprocessed_texts = []
        float_pattern = re.compile(r'\d+\.\d+')
        
        for text in texts:
            doc = self.nlp(text)
            tokens = []
            for token in doc:
                if token.like_num: 
                    if float_pattern.match(token.text):
                        tokens.append(f"FLOAT_{token.text}")
                    else:
                        tokens.append(f"NUM_{token.text}")
                elif not token.is_stop and not token.is_punct:
                    tokens.append(token.lemma_.lower())
            preprocessed_texts.append(' '.join(tokens))
        return preprocessed_texts
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes
            
        # Preprocess texts
        texts = [self.text_fn(node.node) for node in nodes]
        preprocessed_texts = self.preprocess_texts(texts)

        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)

        # Track which nodes to keep
        keep_indices = []
        already_selected = np.zeros(len(nodes), dtype=bool)

        for idx in range(len(nodes)):
            if not already_selected[idx]:
                keep_indices.append(idx)
                already_selected[idx] = True

                # Find similar statements
                similar_indices = np.where(cosine_sim_matrix[idx] > self.similarity_threshold)[0]
                for sim_idx in similar_indices:
                    if not already_selected[sim_idx]:
                        logger.debug(
                            f"Removing duplicate (similarity: {cosine_sim_matrix[idx][sim_idx]:.4f}):\n"
                            f"Kept: {texts[idx]}\n"
                            f"Removed: {texts[sim_idx]}"
                        )
                        already_selected[sim_idx] = True

        return [nodes[i] for i in keep_indices]
