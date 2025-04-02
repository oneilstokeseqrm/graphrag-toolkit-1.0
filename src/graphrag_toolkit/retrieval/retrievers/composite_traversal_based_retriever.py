# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import concurrent.futures
from dataclasses import dataclass
from typing import List, Type, Optional, Union

from graphrag_toolkit.storage.graph import GraphStore
from graphrag_toolkit.storage.vector.vector_store import VectorStore
from graphrag_toolkit.retrieval.retrievers.traversal_based_base_retriever import TraversalBasedBaseRetriever
from graphrag_toolkit.retrieval.utils.query_decomposition import QueryDecomposition
from graphrag_toolkit.retrieval.retrievers.entity_context_search import EntityContextSearch
from graphrag_toolkit.retrieval.retrievers.chunk_based_search import ChunkBasedSearch
from graphrag_toolkit.retrieval.retrievers.keyword_entity_search import KeywordEntitySearch
from graphrag_toolkit.retrieval.processors import *
from graphrag_toolkit.retrieval.model import SearchResultCollection, SearchResult, ScoredEntity, Entity

from llama_index.core.schema import QueryBundle
from llama_index.core.async_utils import run_async_tasks

logger = logging.getLogger(__name__)

TraversalBasedRetrieverType = Union[TraversalBasedBaseRetriever, Type[TraversalBasedBaseRetriever]]

@dataclass
class WeightedTraversalBasedRetriever:
    retriever:TraversalBasedRetrieverType
    weight:float=1.0

DEFAULT_TRAVERSAL_BASED_RETRIEVERS = [
    WeightedTraversalBasedRetriever(retriever=ChunkBasedSearch, weight=0.2), 
    WeightedTraversalBasedRetriever(retriever=EntityContextSearch, weight=1.0)
]

WeightedTraversalBasedRetrieverType = Union[WeightedTraversalBasedRetriever, TraversalBasedBaseRetriever, Type[TraversalBasedBaseRetriever]]

class CompositeTraversalBasedRetriever(TraversalBasedBaseRetriever):

    def __init__(self, 
                 graph_store:GraphStore, 
                 vector_store:VectorStore,
                 retrievers:Optional[List[WeightedTraversalBasedRetrieverType]]=None,
                 query_decomposition:Optional[QueryDecomposition]=None,
                 **kwargs): 

        super().__init__(
            graph_store=graph_store, 
            vector_store=vector_store,
            **kwargs
        )

        self.query_decomposition = query_decomposition or QueryDecomposition(max_subqueries=self.args.max_subqueries)
        self.weighted_retrievers:List[WeightedTraversalBasedRetrieverType] = retrievers or DEFAULT_TRAVERSAL_BASED_RETRIEVERS

    def get_start_node_ids(self, query_bundle: QueryBundle) -> List[str]:
        return []
    
    async def _get_search_results_for_query(self, query_bundle: QueryBundle) -> SearchResultCollection:

        def weighted_arg(v, weight, factor):
            multiplier = min(1, weight * factor)
            return  math.ceil(v * multiplier)

        retrievers = []

        # get entities
        keyword_entity_search = KeywordEntitySearch(
            graph_store=self.graph_store, 
            max_keywords=self.args.max_keywords,
            expand_entities=self.args.expand_entities
        )

        entity_search_results = keyword_entity_search.retrieve(query_bundle)

        entities = [
            ScoredEntity(
                entity=Entity.model_validate_json(entity_search_result.text), 
                score=entity_search_result.score
            )
            for entity_search_result in entity_search_results
        ]

        for wr in self.weighted_retrievers:
            
            if not isinstance(wr, WeightedTraversalBasedRetriever):
                wr = WeightedTraversalBasedRetriever(retriever=wr, weight=1.0)
            
            sub_args = self.args.to_dict()

            sub_args['intermediate_limit'] = weighted_arg(self.args.intermediate_limit, wr.weight, 2)
            sub_args['limit_per_query'] = weighted_arg(self.args.query_limit, wr.weight, 1)

            retriever = (wr.retriever if isinstance(wr.retriever, TraversalBasedBaseRetriever) 
                         else wr.retriever(
                            self.graph_store, 
                            self.vector_store,
                            processors=[
                                # No processing - just raw results
                            ],
                            formatting_processors=[
                                # No processing - just raw results
                            ],
                            entities=entities,
                            **sub_args
                        ))

            retrievers.append(retriever)

        search_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            
            futures = [
                executor.submit(r.aretrieve, query_bundle)
                for r in retrievers
            ]
            
            executor.shutdown()

            for future in futures:
                for scored_node in await future.result():
                    search_results.append(SearchResult.model_validate_json(scored_node.node.text))
        
        return SearchResultCollection(results=search_results, entities=entities)
            
    
    def do_graph_search(self, query_bundle: QueryBundle, start_node_ids:List[str]) -> SearchResultCollection:
        
        search_results = SearchResultCollection()

        subqueries = (self.query_decomposition.decompose_query(query_bundle) 
            if self.args.derive_subqueries 
            else [query_bundle]
        )

        tasks = [
            self._get_search_results_for_query(subquery) 
            for subquery in subqueries
        ]
            
        task_results:List[SearchResultCollection] = run_async_tasks(tasks)

        for task_result in task_results:
            for search_result in task_result.results:
                search_results.add_search_result(search_result) 
            for entity in task_result.entities:
                search_results.add_entity(entity)
        
        return search_results
        