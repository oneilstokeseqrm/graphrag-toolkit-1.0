# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import concurrent.futures
from typing import List, Optional, Type

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection
from graphrag_toolkit.lexical_graph.storage.vector.vector_store import VectorStore
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.retrievers.traversal_based_base_retriever import TraversalBasedBaseRetriever
from graphrag_toolkit.lexical_graph.retrieval.utils.vector_utils import get_diverse_vss_elements

from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import MetadataFilters

logger = logging.getLogger(__name__)

class TopicBasedSearch(TraversalBasedBaseRetriever):
    def __init__(self, 
                 graph_store:GraphStore,
                 vector_store:VectorStore,
                 processor_args:Optional[ProcessorArgs]=None,
                 processors:Optional[List[Type[ProcessorBase]]]=None,
                 filter_config:FilterConfig=None,
                 **kwargs):
        
        super().__init__(
            graph_store=graph_store, 
            vector_store=vector_store,
            processor_args=processor_args,
            processors=processors,
            filter_config=filter_config,
            **kwargs
        )
    
    def topic_based_graph_search(self, topic_id):

        cypher = self.create_cypher_query(f'''
        // topic-based graph search                                  
        MATCH (f:`__Fact__`)-[:`__NEXT__`*0..1]-(:`__Fact__`)-[:`__SUPPORTS__`]->(:`__Statement__`)-[:`__BELONGS_TO__`]->(tt:`__Topic__`)
        WHERE {self.graph_store.node_id("tt.topicId")} = $topicId
        WITH f LIMIT $statementLimit
        MATCH (f)-[:`__SUPPORTS__`]->(:`__Statement__`)-[:`__PREVIOUS__`*0..2]-(l:`__Statement__`)-[:`__BELONGS_TO__`]->(t:`__Topic__`)
        ''')
                                  
        properties = {
            'topicId': topic_id,
            'limit': self.args.query_limit,
            'statementLimit': self.args.intermediate_limit
        }
                                          
        return self.graph_store.execute_query(cypher, properties)

    def get_start_node_ids(self, query_bundle: QueryBundle) -> List[str]:

        logger.debug('Getting start node ids for topic-based search...')

        topics = get_diverse_vss_elements('topic', query_bundle, self.vector_store, self.args, self.filter_config)
        
        return [topic['topic']['topicId'] for topic in topics]
    
    def do_graph_search(self, query_bundle: QueryBundle, start_node_ids:List[str]) -> SearchResultCollection:
        
        topic_ids = start_node_ids

        logger.debug('Running topic-based search...')
        
        search_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:

            futures = [
                executor.submit(self.topic_based_graph_search, topic_id)
                for topic_id in topic_ids
            ]
            
            executor.shutdown()

            for future in futures:
                for result in future.result():
                    search_results.append(result)
                    
        search_results_collection = self._to_search_results_collection(search_results) 
        
        retriever_name = type(self).__name__
        if retriever_name in self.args.debug_results and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'''Topic-based results: {search_results_collection.model_dump_json(
                    indent=2, 
                    exclude_unset=True, 
                    exclude_defaults=True, 
                    exclude_none=True, 
                    warnings=False)
                }''')
                   
        
        return search_results_collection
    
