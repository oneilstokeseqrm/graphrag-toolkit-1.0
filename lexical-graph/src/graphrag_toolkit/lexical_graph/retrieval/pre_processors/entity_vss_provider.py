# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import List, Optional, Dict

from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.utils.tfidf_utils import score_values
from graphrag_toolkit.lexical_graph.retrieval.pre_processors.entity_provider_base import EntityProviderBase
from graphrag_toolkit.lexical_graph.retrieval.pre_processors.entity_provider import EntityProvider
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.post_processors import SentenceReranker

from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode


logger = logging.getLogger(__name__)

class EntityVSSProvider(EntityProviderBase):
    
    def __init__(self, graph_store:GraphStore, vector_store:VectorStore, args:ProcessorArgs, filter_config:Optional[FilterConfig]=None):
        super().__init__(graph_store=graph_store, args=args, filter_config=filter_config)
        self.vector_store = vector_store

        
    def _get_chunk_ids(self, keywords:List[str]) -> List[str]:
        
        query_bundle =  QueryBundle(query_str=', '.join(keywords))
        vss_results = self.vector_store.get_index('chunk').top_k(query_bundle, 3, filter_config=self.filter_config)

        chunk_ids = [result['chunk']['chunkId'] for result in vss_results]

        logger.debug(f'chunk_ids: {chunk_ids}')

        return chunk_ids

    def _get_entities_for_chunks(self, chunk_ids:List[str]) -> List[ScoredEntity]:

        cypher = f"""
        // get entities for chunk ids
        MATCH (c:`__Chunk__`)<-[:`__MENTIONED_IN__`]-(:`__Statement__`)
        <-[:`__SUPPORTS__`]-(:`__Fact__`)<-[:`__SUBJECT__`|`__OBJECT__`]-(entity:`__Entity__`)
        -[r:`__SUBJECT__`]->(f:`__Fact__`)
        WHERE {self.graph_store.node_id("c.chunkId")} in $chunkIds
        WITH DISTINCT entity, count(DISTINCT r) AS score ORDER BY score DESC
        RETURN {{
            {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
            score: score
        }} AS result
        """

        parameters = {
            'chunkIds': chunk_ids
        }

        results = self.graph_store.execute_query(cypher, parameters)

        scored_entities = [
            ScoredEntity.model_validate(result['result'])
            for result in results
            if result['result']['score'] != 0
        ]

        logger.debug(f'entities: {scored_entities}')

        return scored_entities
    
    def _get_reranked_entities(self, entities:List[ScoredEntity], scored_entity_names:Dict[str, float]) -> List[ScoredEntity]:

        logger.debug(f'reranked_entity_names: {scored_entity_names}')

        reranked_entities = []
        entity_id_map = {}

        for scored_entity_name, _ in scored_entity_names.items():
            for entity in entities:
                if entity.entity.value == scored_entity_name and entity.entity.entityId not in entity_id_map:
                    entity_id_map[entity.entity.entityId] = None
                    reranked_entities.append(entity)


        logger.debug(f'reranked_entities: {reranked_entities}')

        reranked_entities = reranked_entities[:self.args.max_vss_entities]

        return reranked_entities
    
    def _get_reranked_entities_model(self, entities:List[ScoredEntity], keywords:List[str]) -> List[ScoredEntity]:

        reranker = SentenceReranker(model=GraphRAGConfig.reranking_model, top_n=len(entities))
        rank_query = QueryBundle(query_str=' '.join(keywords))

        reranked_values = reranker.postprocess_nodes(
            [
                NodeWithScore(node=TextNode(text=entity.entity.value), score=0.0)
                for entity in entities
            ],
            rank_query
        )

        scored_entity_names =  {
            reranked_value.text : reranked_value.score
            for reranked_value in reranked_values
        }

        return self._get_reranked_entities(entities, scored_entity_names)
    
    def _get_reranked_entities_tfidf(self, entities:List[ScoredEntity], keywords:List[str]) -> List[ScoredEntity]:
        
        entity_names = [entity.entity.value for entity in entities]
        scored_entity_names = score_values(entity_names, keywords, ngram_length=1)

        return self._get_reranked_entities(entities, scored_entity_names)

                        
    def get_entities(self, keywords:List[str]) -> List[ScoredEntity]:

        initial_entity_provider = EntityProvider(self.graph_store, self.args, self.filter_config)
        initial_entities = initial_entity_provider.get_entities(keywords)
        
        chunk_ids = self._get_chunk_ids(keywords + [entity.entity.value for entity in initial_entities])
        chunk_entities = self._get_entities_for_chunks(chunk_ids)

        reranked_chunk_entities = []
        if self.args.reranker == 'model':
            reranked_chunk_entities = self._get_reranked_entities_model(chunk_entities, keywords) 
        else:
            reranked_chunk_entities = self._get_reranked_entities_tfidf(chunk_entities, keywords)

        logger.debug(f'initial_entities: {initial_entities}')
        logger.debug(f'reranked_chunk_entities: {reranked_chunk_entities}')
        
        return initial_entities + reranked_chunk_entities

        