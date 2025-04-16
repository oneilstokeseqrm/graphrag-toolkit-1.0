# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.schema import NodeRelationship

from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.build.build_filter import BuildFilter
from graphrag_toolkit.lexical_graph.indexing.build.node_builder import NodeBuilder
from graphrag_toolkit.lexical_graph.indexing.constants import TOPICS_KEY
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

class SourceNodeBuilder(NodeBuilder):
    
    @classmethod
    def name(cls) -> str:
        return 'SourceNodeBuilder'
    
    @classmethod
    def metadata_keys(cls) -> List[str]:
        return [TOPICS_KEY]
    
    def build_nodes(self, nodes:List[BaseNode], filter:BuildFilter):
        
        source_nodes = {}

        for node in nodes:
            
            source_info = node.relationships.get(NodeRelationship.SOURCE, None)
            source_id = source_info.node_id
            
            if source_id not in source_nodes:
                
                metadata = {
                    'source': {
                        'sourceId': source_id
                    }    
                }
                
                if source_info.metadata:
                    metadata['source']['metadata'] = source_info.metadata
                    
                metadata[INDEX_KEY] = {
                    'index': 'source',
                    'key': self._clean_id(source_id)
                }
                
                source_node = TextNode(
                    id_ = source_id,
                    metadata = metadata,
                    excluded_embed_metadata_keys = [INDEX_KEY],
                    excluded_llm_metadata_keys = [INDEX_KEY]
                )

                source_nodes[source_id] = source_node

        return list(source_nodes.values())


