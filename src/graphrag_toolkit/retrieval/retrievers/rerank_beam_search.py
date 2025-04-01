# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import PriorityQueue
from typing import List, Dict, Set, Tuple, Optional, Any, Union, Type

from graphrag_toolkit.storage.graph import GraphStore
from graphrag_toolkit.storage import VectorStore
from graphrag_toolkit.retrieval.utils.statement_utils import get_statements_query
from graphrag_toolkit.retrieval.retrievers.semantic_guided_base_retriever import SemanticGuidedBaseRetriever
from graphrag_toolkit.retrieval.post_processors import RerankerMixin

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)

class RerankingBeamGraphSearch(SemanticGuidedBaseRetriever):
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        reranker: RerankerMixin,
        initial_retrievers: Optional[List[Union[SemanticGuidedBaseRetriever, Type[SemanticGuidedBaseRetriever]]]] = None,
        shared_nodes: Optional[List[NodeWithScore]] = None,
        max_depth: int = 3,
        beam_width: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(vector_store, graph_store, **kwargs)
        self.reranker = reranker 
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.shared_nodes = shared_nodes
        self.score_cache = {}
        self.statement_cache = {} 

        # Initialize initial retrievers if provided
        self.initial_retrievers = []
        if initial_retrievers:
            for retriever in initial_retrievers:
                if isinstance(retriever, type):
                    self.initial_retrievers.append(
                        retriever(vector_store, graph_store, **kwargs)
                    )
                else:
                    self.initial_retrievers.append(retriever)


    def get_statements(self, statement_ids: List[str]) -> Dict[str, Dict]:
        """Fetch statements, using cache when possible."""
        uncached_ids = [sid for sid in statement_ids if sid not in self.statement_cache]
        if uncached_ids:
            new_results = get_statements_query(self.graph_store, uncached_ids)
            for result in new_results:
                sid = result['result']['statement']['statementId']
                self.statement_cache[sid] = result['result']
        
        return {sid: self.statement_cache[sid] for sid in statement_ids}
        
    def get_neighbors(self, statement_id: str) -> List[str]:
        """Get neighboring statements through entity connections."""
        cypher = f"""
        MATCH (e:`__Entity__`)-[:`__SUBJECT__`|`__OBJECT__`]->(:`__Fact__`)-[:`__SUPPORTS__`]->(s:`__Statement__`)
        WHERE {self.graph_store.node_id('s.statementId')} = $statementId
        WITH s, COLLECT(DISTINCT e) AS entities
        UNWIND entities AS entity
        MATCH (entity)-[:`__SUBJECT__`|`__OBJECT__`]->(:`__Fact__`)-[:`__SUPPORTS__`]->(e_neighbors:`__Statement__`)
        RETURN DISTINCT {self.graph_store.node_id('e_neighbors.statementId')} as statementId
        """
        
        neighbors = self.graph_store.execute_query(
            cypher, 
            {'statementId': statement_id}
        )
        return [n['statementId'] for n in neighbors]
    
    def rerank_statements(
        self,
        query: str,
        statement_ids: List[str],
        statement_texts: Dict[str, str]
    ) -> List[Tuple[float, str]]:
        """Rerank statements using the provided reranker."""
        uncached_statements = [statement_texts[sid] for sid in statement_ids if statement_texts[sid] not in self.score_cache]
        
        if uncached_statements:
            pairs = [
                (query, statement_text)
                for statement_text in uncached_statements
            ]

            scores = self.reranker.rerank_pairs(
                pairs=pairs,
                batch_size=self.reranker.batch_size*2
            )

            for statement_text, score in zip(uncached_statements, scores):
                self.score_cache[statement_text] = score
            
        scored_pairs = []
        for sid in statement_ids:
            score = self.score_cache[statement_texts[sid]]
            scored_pairs.append(
                (score, sid)
            )

        scored_pairs.sort(reverse=True)
        return scored_pairs

    def beam_search(
        self, 
        query_bundle: QueryBundle,
        start_statement_ids: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """Perform beam search using reranker for scoring."""
        visited: Set[str] = set()
        results: List[Tuple[str, List[str]]] = []
        queue: PriorityQueue = PriorityQueue()

        # Get texts for all start statements
        start_statements = self.get_statements(start_statement_ids)
        statement_texts = {
            sid: statement['statement']['value']
            for sid, statement in start_statements.items()
        }

        # Score initial statements using reranker
        start_scores = self.rerank_statements(
            query_bundle.query_str,
            start_statement_ids,
            statement_texts
        )

        # Initialize queue with start statements
        for score, statement_id in start_scores:
            queue.put((-score, 0, statement_id, [statement_id]))

        while not queue.empty() and len(results) < self.beam_width:
            neg_score, depth, current_id, path = queue.get()

            if current_id in visited:
                continue

            visited.add(current_id)
            results.append((current_id, path))

            if depth < self.max_depth:
                # Get and score neighbors
                neighbor_ids = self.get_neighbors(current_id)
                if neighbor_ids:
                    # Get texts for neighbors
                    neighbor_statements = self.get_statements(neighbor_ids)
                    neighbor_texts = {
                        sid: str(statement['statement']['value']+'\n'+statement['statement']['details'])
                        for sid, statement in neighbor_statements.items()
                    }

                    # Score neighbors using reranker
                    scored_neighbors = self.rerank_statements(
                        query_bundle.query_str,
                        neighbor_ids,
                        neighbor_texts
                    )

                    # Add top-k neighbors to queue
                    for score, neighbor_id in scored_neighbors[:self.beam_width]:
                        if neighbor_id not in visited:
                            new_path = path + [neighbor_id]
                            queue.put((-score, depth + 1, neighbor_id, new_path))

        return results
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve statements using beam search."""
        try:
            # Get initial nodes (either shared or from initial retrievers)
            initial_statement_ids = set()
            
            if self.shared_nodes is not None:
                # Use shared nodes if available
                for node in self.shared_nodes:
                    initial_statement_ids.add(
                        node.node.metadata['statement']['statementId']
                    )
            elif self.initial_retrievers:
                # Get nodes from initial retrievers
                for retriever in self.initial_retrievers:
                    nodes = retriever.retrieve(query_bundle)
                    for node in nodes:
                        initial_statement_ids.add(
                            node.node.metadata['statement']['statementId']
                        )
            else:
                # Fallback to vector similarity if no initial nodes
                results = self.vector_store.get_index('statement').top_k(
                    query_bundle,
                    top_k=self.beam_width * 2
                )
                initial_statement_ids = {
                    r['statement']['statementId'] for r in results
                }

            if not initial_statement_ids:
                logger.warning("No initial statements found for the query.")
                return []

            # Perform beam search
            beam_results = self.beam_search(
                query_bundle,
                list(initial_statement_ids)
            )

            # Collect all new statement IDs from beam search
            new_statement_ids = [
                statement_id for statement_id, _ in beam_results
                if statement_id not in initial_statement_ids
            ]

            if not new_statement_ids:
                logger.info("Beam search did not find any new statements.")
                return []

            # Create nodes from results
            nodes = []
            statement_to_path = {
                statement_id: path 
                for statement_id, path in beam_results 
                if statement_id not in initial_statement_ids
            }
            
            for statement_id, path in statement_to_path.items():
                statement_data = self.statement_cache.get(statement_id)
                if statement_data:
                    node = TextNode(
                        text=statement_data['statement']['value'],
                        metadata={
                            'statement': statement_data['statement'],
                            'chunk': statement_data['chunk'],
                            'source': statement_data['source'],
                            'search_type': 'beam_search',
                            'depth': len(path),
                            'path': path
                        }
                    )
                    score = self.score_cache.get(statement_data['statement']['value'], 0.0)
                    nodes.append(NodeWithScore(node=node, score=score))
                else:
                    logger.warning(f"Statement data not found in cache for ID: {statement_id}")

            nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

            logger.info(f"Retrieved {len(nodes)} new nodes through beam search.")
            return nodes

        except Exception as e:
            logger.error(f"Error in _retrieve: {str(e)}")
            return []