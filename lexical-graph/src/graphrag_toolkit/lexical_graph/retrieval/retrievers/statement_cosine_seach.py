# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any, Optional

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.utils.statement_utils import get_top_k, SharedEmbeddingCache
from graphrag_toolkit.lexical_graph.retrieval.retrievers.semantic_guided_base_retriever import SemanticGuidedBaseRetriever

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)

class StatementCosineSimilaritySearch(SemanticGuidedBaseRetriever):
    """Retrieves statements using cosine similarity of embeddings."""

    def __init__(
        self,
        vector_store:VectorStore,
        graph_store:GraphStore,
        embedding_cache:Optional[SharedEmbeddingCache]=None,
        top_k:int=100,
        filter_config:Optional[FilterConfig]=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vector_store, graph_store, filter_config, **kwargs)
        self.embedding_cache = embedding_cache
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    
        # 1. Get initial candidates from vector store via L2 Norm
        statement_results = self.vector_store.get_index('statement').top_k(
            query_bundle, 
            top_k=500,
            filter_config=self.filter_config
        )
        
        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'statement_results: {statement_results}')
        else:
            logger.debug(f'num statement_results: {len(statement_results)}')
        
        # 2. Get statement IDs and embeddings using shared cache
        statement_ids = [r['statement']['statementId'] for r in statement_results]
        statement_embeddings = self.embedding_cache.get_embeddings(statement_ids)

        # 3. Get top-k statements by cosine similarity
        top_k_statements = get_top_k(
            query_bundle.embedding,
            statement_embeddings,
            self.top_k
        )
        
        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'top_k_statements: {top_k_statements}')
        else:
            logger.debug(f'num top_k_statements: {len(top_k_statements)}')

        # 4. Create nodes with minimal data
        nodes = []
        for score, statement_id in top_k_statements:
            node = TextNode(
                text="",  # Placeholder - will be populated by StatementGraphRetriever
                metadata={
                    'statement': {'statementId': statement_id},
                    'search_type': 'cosine_similarity'
                }
            )
            nodes.append(NodeWithScore(node=node, score=score))

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results: 
            logger.debug(f'nodes: {nodes}')
        else:
            logger.debug(f'num nodes: {len(nodes)}')

        return nodes
