# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict, Any, Optional

from graphrag_toolkit import TenantId
from graphrag_toolkit.storage.constants import LEXICAL_GRAPH_LABELS
from graphrag_toolkit.storage.graph_store import GraphStore, NodeId

class MultiTenantGraphStore(GraphStore):

    @classmethod
    def wrap(cls, graph_store:GraphStore, tenant_id:TenantId, labels:List[str]=LEXICAL_GRAPH_LABELS):
        if tenant_id.is_default_tenant():
            return graph_store
        if isinstance(graph_store, MultiTenantGraphStore):
            return graph_store
        return MultiTenantGraphStore(inner=graph_store, tenant_id=tenant_id, labels=labels)
    
    inner:GraphStore
    labels:List[str]=[]

    def execute_query_with_retry(self, query:str, parameters:Dict[str, Any], max_attempts=3, max_wait=5, **kwargs):
        self.inner.execute_query_with_retry(query=self._rewrite_query(query), parameters=parameters, max_attempts=max_attempts, max_wait=max_wait)

    def _logging_prefix(self, query_id:str, correlation_id:Optional[str]=None):
        return self.inner._logging_prefix(query_id=query_id, correlation_id=correlation_id) 
    
    def node_id(self, id_name:str) -> NodeId:
        return self.inner.node_id(id_name=id_name)
    
    def execute_query(self, cypher:str, parameters={}, correlation_id=None) -> Dict[str, Any]:
        return self.inner.execute_query(cypher=self._rewrite_query(cypher), parameters=parameters, correlation_id=correlation_id)
    
    def _rewrite_query(self, cypher:str):
        if self.tenant_id.is_default_tenant():
            return cypher
        for label in self.labels:
            original_label = f'`{label}`'
            new_label = self.tenant_id.format_label(label)
            cypher = cypher.replace(original_label, new_label)
        return cypher
    