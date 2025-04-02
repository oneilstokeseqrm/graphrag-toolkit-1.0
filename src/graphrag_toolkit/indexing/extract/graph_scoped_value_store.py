# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List
from graphrag_toolkit.indexing.extract.scoped_value_provider import ScopedValueStore
from graphrag_toolkit.storage.graph import GraphStore

logger = logging.getLogger(__name__)

class GraphScopedValueStore(ScopedValueStore):
    
    graph_store: GraphStore

    def get_scoped_values(self, label:str, scope:str) -> List[str]:
        
        cypher = f'''
        MATCH (n:`__SYS_SV__{label}__`)
        WHERE n.scope=$scope
        RETURN DISTINCT n.value AS value
        '''

        params = {
            'scope': scope
        }

        results = self.graph_store.execute_query(cypher, params)

        return [result['value'] for result in results]

    def save_scoped_values(self, label:str, scope:str, values:List[str]) -> None:
        
        cypher = f'''
        UNWIND $values AS value
        MERGE (:`__SYS_SV__{label}__`{{scope:$scope, value:value}})
        '''

        params = {
            'scope': scope,
            'values': values
        }

        self.graph_store.execute_query_with_retry(cypher, params)

   

    

