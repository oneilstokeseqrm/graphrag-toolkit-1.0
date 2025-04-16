# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, List
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore

class GraphBatchClient():
    def __init__(self, graph_client:GraphStore, batch_writes_enabled:bool, batch_write_size:int):
        self.graph_client = graph_client
        self.batch_writes_enabled = batch_writes_enabled
        self.batch_write_size = batch_write_size
        self.batches = {}
        self.all_nodes = []

    @property
    def tenant_id(self):
        return self.graph_client.tenant_id

    def node_id(self, id_name:str):
        return self.graph_client.node_id(id_name)
    
    def execute_query_with_retry(self, query:str, properties:Dict[str, Any], **kwargs):
        if not self.batch_writes_enabled:
            self.graph_client.execute_query_with_retry(query, properties, **kwargs)
        else:
            if query not in self.batches:
                self.batches[query] = []
            self.batches[query].extend(properties['params'])

    def allow_yield(self, node):
        if self.batch_writes_enabled:
            self.all_nodes.append(node)
            return False
        else:
            return True
        
    def apply_batch_operations(self):
        for query, parameters in self.batches.items():

            deduped_parameters = self._dedup(parameters)
            parameter_chunks = [
                deduped_parameters[x:x+self.batch_write_size] 
                for x in range(0, len(deduped_parameters), self.batch_write_size)
            ]

            for p in parameter_chunks:
                params = {
                    'params': p
                }
                self.graph_client.execute_query_with_retry(query, params, max_attempts=5, max_wait=7)

        return self.all_nodes
  
    def _dedup(self, parameters:List):
        params_map = {}
        for p in parameters:
            params_map[str(p).lower()] = p
        return list(params_map.values())
    
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

    