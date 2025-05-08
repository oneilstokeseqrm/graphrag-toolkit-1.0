# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any
from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.build.build_filters import BuildFilters
from graphrag_toolkit.lexical_graph.indexing.build.node_builder import NodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.source_node_builder import SourceNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.chunk_node_builder import ChunkNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.topic_node_builder import TopicNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.statement_node_builder import StatementNodeBuilder

from llama_index.core.schema import BaseNode, NodeRelationship

logger = logging.getLogger(__name__)

class NodeBuilders():

    def __init__(self, builders:List[NodeBuilder]=[], filters:BuildFilters=None, id_generator:IdGenerator=None):

        id_generator = id_generator or IdGenerator()

        self.builders = builders or self.default_builders(id_generator)
        self.filters = filters or BuildFilters()
        self.id_generator = id_generator

        logger.debug(f'Node builders: {[type(b).__name__ for b in self.builders]}')
    
    def default_builders(self, id_generator:IdGenerator):
        return [
            SourceNodeBuilder(id_generator=id_generator),
            ChunkNodeBuilder(id_generator=id_generator),
            TopicNodeBuilder(id_generator=id_generator),
            StatementNodeBuilder(id_generator=id_generator)
        ]
        
    @classmethod
    def class_name(cls) -> str:
        return 'NodeBuilders'
    
    def get_nodes_from_metadata(self, input_nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        
        def apply_tenant_rewrites(node):
            
            node.id_ =  self.id_generator.rewrite_id_for_tenant(node.id_)

            node_relationships = {}

            for rel, node_info in node.relationships.items():
                if isinstance(node_info, list):
                    node_info_list = []
                    for n in node_info:
                        n.node_id = self.id_generator.rewrite_id_for_tenant(n.node_id) 
                        node_info_list.append(n)
                    node_relationships[rel] = node_info_list
                else:
                    node_info.node_id = self.id_generator.rewrite_id_for_tenant(node_info.node_id)
                    node_relationships[rel] = node_info
           
            return node
        
        def clean_text(node):
            node.text = node.text.replace('\x00', '')
            return node
        
        def pre_process(node):
            node = clean_text(node)
            node = apply_tenant_rewrites(node)
            return node

        results = []

        filtered_nodes = [
            node 
            for node in input_nodes 
            if self.filters.filter_source_metadata_dictionary(node.relationships[NodeRelationship.SOURCE].metadata) 
        ]

        pre_processed_nodes = [
            pre_process(node) 
            for node in filtered_nodes
        ]

        for builder in self.builders:
            try:
                
                builder_specific_nodes = [
                    node
                    for node in pre_processed_nodes 
                    if any(key in builder.metadata_keys() for key in node.metadata)
                ]
                
                results.extend(builder.build_nodes(builder_specific_nodes, self.filters))
            except Exception as e:
                    logger.exception('An error occurred while building nodes from chunks')
                    raise e
            
        results.extend(input_nodes) # Always add the original nodes after derived nodes    

        logger.debug(f'Accepted {len(input_nodes)} chunks, emitting {len(results)} nodes')

        return results
        
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:    
        return self.get_nodes_from_metadata(nodes, **kwargs)
                    
