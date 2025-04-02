# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.indexing.model import Fact
from graphrag_toolkit.storage.graph import GraphStore
from graphrag_toolkit.storage.graph.graph_utils import search_string_from, label_from, relationship_name_from
from graphrag_toolkit.indexing.build.graph_builder import GraphBuilder
from graphrag_toolkit.indexing.constants import DEFAULT_CLASSIFICATION

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class EntityRelationGraphBuilder(GraphBuilder):
    
    @classmethod
    def index_key(cls) -> str:
        return 'fact'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
            
        fact_metadata = node.metadata.get('fact', {})
        include_domain_labels = kwargs['include_domain_labels']

        if fact_metadata:

            fact = Fact.model_validate(fact_metadata)

            if fact.subject and fact.object:
        
                logger.debug(f'Inserting entity relations for fact [fact_id: {fact.factId}]')

                statements = [
                    '// insert entity relations',
                    'UNWIND $params AS params'
                ]

                if include_domain_labels:
                    statements.append(f'MERGE (subject:`__Entity__`:{label_from(fact.subject.classification or DEFAULT_CLASSIFICATION)}{{{graph_client.node_id("entityId")}: params.s_id}})')
                    statements.append(f'MERGE (object:`__Entity__`:{label_from(fact.object.classification or DEFAULT_CLASSIFICATION)}{{{graph_client.node_id("entityId")}: params.o_id}})')
                else:
                    statements.append(f'MERGE (subject:`__Entity__`{{{graph_client.node_id("entityId")}: params.s_id}})')
                    statements.append(f'MERGE (object:`__Entity__`{{{graph_client.node_id("entityId")}: params.o_id}})')

                statements.extend([
                    'MERGE (subject)-[r:`__RELATION__`{value: params.p}]->(object)',
                    'ON CREATE SET r.count = 1 ON MATCH SET r.count = r.count + 1'
                ])

                if include_domain_labels:
                    statements.extend([
                        f'MERGE (subject)-[rr:`{relationship_name_from(fact.predicate.value)}`]->(object)',
                        'ON CREATE SET rr.count = 1 ON MATCH SET rr.count = rr.count + 1'
                    ])


                properties = {
                    's_id': fact.subject.entityId,
                    'o_id': fact.object.entityId,
                    'p': fact.predicate.value
                }
            
                query = '\n'.join(statements)
                    
                graph_client.execute_query_with_retry(query, self._to_params(properties), max_attempts=5, max_wait=7)

            else:
                logger.debug(f'SPC fact, so not creating relation [fact_id: {fact.factId}]')
           

        else:
            logger.warning(f'fact_id missing from fact node [node_id: {node.node_id}]')