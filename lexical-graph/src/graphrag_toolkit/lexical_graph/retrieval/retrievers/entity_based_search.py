# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import concurrent.futures
from typing import List, Generator, Tuple, Any, Optional, Type

from graphrag_toolkit.lexical_graph import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, ScoredEntity, Entity
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector.vector_store import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.retrievers.keyword_entity_search import KeywordEntitySearch
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.retrievers.traversal_based_base_retriever import TraversalBasedBaseRetriever

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

class EntityBasedSearch(TraversalBasedBaseRetriever):
    def __init__(self, 
                 graph_store:GraphStore,
                 vector_store:VectorStore,
                 processor_args:Optional[ProcessorArgs]=None,
                 processors:Optional[List[Type[ProcessorBase]]]=None,
                 filter_config:Optional[FilterConfig]=None,
                 **kwargs):
        
        super().__init__(
            graph_store=graph_store, 
            vector_store=vector_store,
            processor_args=processor_args,
            processors=processors,
            filter_config=filter_config,
            **kwargs
        )

    def get_start_node_ids(self, query_bundle: QueryBundle) -> List[str]:

        logger.debug('Getting start node ids for entity-based search...')

        if self.entities:
            return [entity.entity.entityId for entity in self.entities]
        
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

        return [entity.entity.entityId for entity in entities]
    
    def _for_each_disjoint(self, values:List[Any], others:Optional[List[Any]]=None) -> Generator[Tuple[Any, List[Any]], None, None]:
        values_as_set = set(values)
        for value in values:
            other_values = others or list(values_as_set.difference({value}))
            yield (value, other_values)
            
    def _for_each_disjoint_unique(self, values:List[Any]) -> Generator[Tuple[Any, List[Any]], None, None]:
        for idx, value in enumerate(values[:-1]):
            other_values = values[idx+1:]
            yield (value, other_values)

    
    def _multiple_entity_based_graph_search(self, start_id, end_ids, query:QueryBundle):

        logger.debug(f'Starting multiple-entity-based searches for [start_id: {start_id}, end_ids: {end_ids}]')
        
        cypher = self.create_cypher_query(f''' 
        // multiple entity-based graph search                                                                
        MATCH p=(e1:`__Entity__`{{{self.graph_store.node_id("entityId")}:$startId}})-[:`__RELATION__`*1..2]-(e2:`__Entity__`) 
        WHERE {self.graph_store.node_id("e2.entityId")} in $endIds
        UNWIND nodes(p) AS n
        WITH DISTINCT COLLECT(n) AS entities
        MATCH (s:`__Entity__`)-[:`__SUBJECT__`]->(f:`__Fact__`)<-[:`__OBJECT__`]-(o:`__Entity__`),
            (f)-[:`__SUPPORTS__`]->(:`__Statement__`)
            -[:`__PREVIOUS__`*0..1]-(l:`__Statement__`)
            -[:`__BELONGS_TO__`]->(t:`__Topic__`)
        WHERE s in entities and o in entities
        ''')
            
        properties = {
            'startId': start_id,
            'endIds': end_ids,
            'statementLimit': self.args.intermediate_limit,
            'limit': self.args.query_limit
        }
            
        return self.graph_store.execute_query(cypher, properties)
           

    def _single_entity_based_graph_search(self, entity_id, query:QueryBundle):

        logger.debug(f'Starting single-entity-based search for [entity_id: {entity_id}]')
            
        cypher = self.create_cypher_query(f''' 
        // single entity-based graph search                            
        MATCH (:`__Entity__`{{{self.graph_store.node_id("entityId")}:$startId}})
            -[:`__SUBJECT__`]->(f:`__Fact__`)
            -[:`__SUPPORTS__`]->(:`__Statement__`)
            -[:`__PREVIOUS__`*0..1]-(l:`__Statement__`)
            -[:`__BELONGS_TO__`]->(t:`__Topic__`)''')
            
        properties = {
            'startId': entity_id,
            'statementLimit': self.args.intermediate_limit,
            'limit': self.args.query_limit
        }
            
        return self.graph_store.execute_query(cypher, properties)
            
    
    def do_graph_search(self, query_bundle:QueryBundle, start_node_ids:List[str]) -> SearchResultCollection:

        logger.debug('Running entity-based search...')
        
        search_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            
            futures = [
                executor.submit(self._multiple_entity_based_graph_search, start_id, end_ids, query_bundle)
                for (start_id, end_ids) in self._for_each_disjoint(start_node_ids)
            ]
            
            futures.extend([
                executor.submit(self._single_entity_based_graph_search, entity_id, query_bundle)
                for entity_id in start_node_ids
            ])
            
            executor.shutdown()

            for future in futures:
                for result in future.result():
                    search_results.append(result)
                    
                
        search_results_collection = self._to_search_results_collection(search_results) 
        
        retriever_name = type(self).__name__
        if retriever_name in self.args.debug_results and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'''Entity-based results: {search_results_collection.model_dump_json(
                    indent=2, 
                    exclude_unset=True, 
                    exclude_defaults=True, 
                    exclude_none=True, 
                    warnings=False)
                }''')
                   
        
        return search_results_collection