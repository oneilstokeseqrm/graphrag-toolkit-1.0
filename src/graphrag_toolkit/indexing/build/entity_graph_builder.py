# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.indexing.model import Fact
from graphrag_toolkit.storage.graph import GraphStore
from graphrag_toolkit.storage.graph.graph_utils import search_string_from, label_from
from graphrag_toolkit.indexing.build.graph_builder import GraphBuilder
from graphrag_toolkit.indexing.constants import DEFAULT_CLASSIFICATION

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class EntityGraphBuilder(GraphBuilder):
    
    @classmethod
    def index_key(cls) -> str:
        return 'fact'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
            
        fact_metadata = node.metadata.get('fact', {})
        include_domain_labels = kwargs['include_domain_labels']

        if fact_metadata:

            fact = Fact.model_validate(fact_metadata)
        
            logger.debug(f'Inserting entities for fact [fact_id: {fact.factId}]')

            statements = [
                '// insert entities',
                'UNWIND $params AS params'
            ]

            if include_domain_labels:
                statements.append(f'MERGE (subject:`__Entity__`:{label_from(fact.subject.classification or DEFAULT_CLASSIFICATION)}{{{graph_client.node_id("entityId")}: params.s_id}})')
            else:
                statements.append(f'MERGE (subject:`__Entity__`{{{graph_client.node_id("entityId")}: params.s_id}})')

            statements.extend([
                'ON CREATE SET subject.value = params.s, subject.search_str = params.s_search_str, subject.class = params.sc',
                'ON MATCH SET subject.value = params.s, subject.search_str = params.s_search_str, subject.class = params.sc',
            ])

            properties = {
                's_id': fact.subject.entityId,
                's': fact.subject.value,
                's_search_str': search_string_from(fact.subject.value),
                'sc': fact.subject.classification or DEFAULT_CLASSIFICATION
            }

            if fact.object and fact.object.entityId != fact.subject.entityId:

                if include_domain_labels:
                    statements.append(f'MERGE (object:`__Entity__`:{label_from(fact.object.classification or DEFAULT_CLASSIFICATION)}{{{graph_client.node_id("entityId")}: params.o_id}})')
                else:
                    statements.append(f'MERGE (object:`__Entity__`{{{graph_client.node_id("entityId")}: params.o_id}})')

                statements.extend([
                    'ON CREATE SET object.value = params.o, object.search_str = params.o_search_str, object.class = params.oc',
                    'ON MATCH SET object.value = params.o, object.search_str = params.o_search_str, object.class = params.oc', 
                ])

                properties.update({                
                    'o_id': fact.object.entityId,
                    'o': fact.object.value,
                    'o_search_str': search_string_from(fact.object.value),
                    'oc': fact.object.classification or DEFAULT_CLASSIFICATION
                })
        
            query = '\n'.join(statements)
                
            graph_client.execute_query_with_retry(query, self._to_params(properties), max_attempts=5, max_wait=7)
           

        else:
            logger.warning(f'fact_id missing from fact node [node_id: {node.node_id}]')