# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.indexing.model import Fact
from graphrag_toolkit.storage.graph_store import GraphStore
from graphrag_toolkit.storage.graph_utils import label_from, relationship_name_from
from graphrag_toolkit.indexing.build.graph_builder import GraphBuilder
from graphrag_toolkit.indexing.constants import DEFAULT_CLASSIFICATION

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class GraphSummaryBuilder(GraphBuilder):
    
    @classmethod
    def index_key(cls) -> str:
        return 'fact'
    
    def build(self, node:BaseNode, graph_client:GraphStore, **kwargs:Any):
            
        fact_metadata = node.metadata.get('fact', {})
        
        if fact_metadata:

            fact = Fact.model_validate(fact_metadata)

            if fact.subject and fact.object:

                tenant_id = graph_client.tenant_id if graph_client.tenant_id else ''

                sc_id = f'sys_class:{tenant_id}:{fact.subject.classification or DEFAULT_CLASSIFICATION}'
                oc_id = f'sys_class:{tenant_id}:{fact.object.classification or DEFAULT_CLASSIFICATION}'

                statements = []

                if sc_id == oc_id:

                    statements.extend( [
                        '// insert graph summary',
                        'UNWIND $params AS params',
                        f'MERGE (sc:`__SYS_Class__`{{{graph_client.node_id("sysClassId")}: params.sc_id}})',
                        'ON CREATE SET sc.value = params.sc, sc.count = 2 ON MATCH SET sc.count = sc.count + 2',                       
                        'MERGE (sc)-[r:`__SYS_RELATION__`{value: params.p}]->(sc)',
                        'ON CREATE SET r.count = 2 ON MATCH SET r.count = r.count + 2'                      
                    ])

                else:

                    statements.extend( [
                        '// insert graph summary',
                        'UNWIND $params AS params',
                        f'MERGE (sc:`__SYS_Class__`{{{graph_client.node_id("sysClassId")}: params.sc_id}})',
                        'ON CREATE SET sc.value = params.sc, sc.count = 1 ON MATCH SET sc.count = sc.count + 1',
                        f'MERGE (oc:`__SYS_Class__`{{{graph_client.node_id("sysClassId")}: params.oc_id}})',
                        'ON CREATE SET oc.value = params.oc, oc.count = 1 ON MATCH SET oc.count = oc.count + 1',
                        'MERGE (sc)-[r:`__SYS_RELATION__`{value: params.p}]->(oc)',
                        'ON CREATE SET r.count = 1 ON MATCH SET r.count = r.count + 1'
                        
                    ])

                properties = {
                    'sc_id': sc_id,
                    'oc_id': oc_id,
                    'sc': label_from(fact.subject.classification or DEFAULT_CLASSIFICATION),
                    'oc': label_from(fact.object.classification or DEFAULT_CLASSIFICATION),
                    'p': relationship_name_from(fact.predicate.value),
                }

                query = '\n'.join(statements)
                    
                graph_client.execute_query_with_retry(query, self._to_params(properties), max_attempts=5, max_wait=7)

        else:
            logger.warning(f'fact_id missing from fact node [node_id: {node.node_id}]')