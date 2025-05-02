# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.indexing.build.graph_builder import GraphBuilder

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class SourceGraphBuilder(GraphBuilder):
    
    @classmethod
    def index_key(cls) -> str:
        return 'source'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
            
        source_metadata = node.metadata.get('source', {})
        source_id = source_metadata.get('sourceId', None)

        if source_id:

            logger.debug(f'Inserting source [source_id: {source_id}]')
        
            statements = [
                '// insert source',
                'UNWIND $params AS params',
                f"MERGE (source:`__Source__`{{{graph_client.node_id('sourceId')}: '{source_id}'}})"
            ]

            metadata = source_metadata.get('metadata', {})
            
            clean_metadata = {}
            metadata_assignments_fns = {}

            for k, v in metadata.items():
                key = k.replace(' ', '_')
                value = str(v)
                clean_metadata[key] = value
                metadata_assignments_fns[key] = graph_client.property_assigment_fn(key, value)

            def format_assigment(key):
                assigment = f'params.{key}'
                return metadata_assignments_fns[key](assigment)
        
            if clean_metadata:
                all_properties = ', '.join(f'source.{key} = {format_assigment(key)}' for key,_ in clean_metadata.items())
                statements.append(f'ON CREATE SET {all_properties} ON MATCH SET {all_properties}')
            
            query = '\n'.join(statements)
            
            graph_client.execute_query_with_retry(query, self._to_params(clean_metadata))

        else:
            logger.warning(f'source_id missing from source node [node_id: {node.node_id}]')