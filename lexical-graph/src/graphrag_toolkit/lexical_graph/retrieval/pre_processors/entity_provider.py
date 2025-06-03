# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import logging
from itertools import repeat
from typing import List, Iterator, cast, Optional

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result, search_string_from
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.retrieval.prompts import SIMPLE_EXTRACT_KEYWORDS_PROMPT, EXTENDED_EXTRACT_KEYWORDS_PROMPT

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)

class EntityProvider():
    
    def __init__(self, graph_store:GraphStore):
        
        self.graph_store = graph_store

        
    def _get_entities_for_keyword(self, keyword:str) -> List[ScoredEntity]:

        parts = keyword.split('|')

        if len(parts) > 1:

            cypher = f"""
            // get entities for keywords
            MATCH (entity:`__Entity__`)-[r:`__SUBJECT__`|`__OBJECT__`]->(:`__Fact__`)
            WHERE entity.search_str = $keyword and entity.class STARTS WITH $classification
            WITH entity, count(r) AS score ORDER BY score DESC
            RETURN {{
                {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                score: score
            }} AS result"""

            params = {
                'keyword': search_string_from(parts[0]),
                'classification': parts[1]
            }
        else:
            cypher = f"""
            // get entities for keywords
            MATCH (entity:`__Entity__`)-[r:`__SUBJECT__`|`__OBJECT__`]->(:`__Fact__`)
            WHERE entity.search_str = $keyword
            WITH entity, count(r) AS score ORDER BY score DESC
            RETURN {{
                {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                score: score
            }} AS result"""

            params = {
                'keyword': search_string_from(parts[0])
            }

        results = self.graph_store.execute_query(cypher, params)

        return [
            ScoredEntity.model_validate(result['result'])
            for result in results
            if result['result']['score'] != 0
        ]
                        
    def get_entities(self, keywords:List[str])  -> List[ScoredEntity]:

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(keywords)) as p:
            scored_entity_batches:Iterator[List[ScoredEntity]] = p.map(self._get_entities_for_keyword, keywords)
            scored_entities = sum(scored_entity_batches, start=cast(List[ScoredEntity], []))

        scored_entity_mappings = {}
        
        for scored_entity in scored_entities:
            entity_id = scored_entity.entity.entityId
            if entity_id not in scored_entity_mappings:
                scored_entity_mappings[entity_id] = scored_entity
            else:
                scored_entity_mappings[entity_id].score += scored_entity.score

        scored_entities = list(scored_entity_mappings.values())

        scored_entities.sort(key=lambda e:e.score, reverse=True)

        logger.debug('Entities:\n' + '\n'.join(
            entity.model_dump_json(exclude_unset=True, exclude_defaults=True, exclude_none=True, warnings=False) 
            for entity in scored_entities)
        )

        return scored_entities

        