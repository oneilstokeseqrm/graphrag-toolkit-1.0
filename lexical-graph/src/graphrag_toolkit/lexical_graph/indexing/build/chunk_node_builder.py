# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from llama_index.core.schema import BaseNode, DEFAULT_TEXT_NODE_TMPL
from llama_index.core.schema import NodeRelationship

from graphrag_toolkit.lexical_graph.indexing.build.build_filters import BuildFilters
from graphrag_toolkit.lexical_graph.indexing.build.node_builder import NodeBuilder
from graphrag_toolkit.lexical_graph.indexing.constants import TOPICS_KEY
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

class ChunkNodeBuilder(NodeBuilder):
    
    @classmethod
    def name(cls) -> str:
        return 'ChunkNodeBuilder'
    
    @classmethod
    def metadata_keys(cls) -> List[str]:
        return [TOPICS_KEY]
    
    def build_nodes(self, nodes:List[BaseNode]):

        chunk_nodes = []

        for node in nodes:

            chunk_id = node.node_id
            chunk_node = node.model_copy()
            chunk_node.text_template = DEFAULT_TEXT_NODE_TMPL
            
            topics = [
                topic['value'] 
                for topic in node.metadata.get(TOPICS_KEY, {}).get('topics', []) 
                if not self.build_filters.ignore_topic(topic['value'])
            ]

            source_info = node.relationships[NodeRelationship.SOURCE]
            source_id = source_info.node_id

            metadata = {
                'source': {
                     'sourceId': source_id
                },
                'chunk': {
                    'chunkId': chunk_id  
                },
                'topics': topics 
            }  
                
            if source_info.metadata:
                metadata['source']['metadata'] = source_info.metadata
            
            metadata[INDEX_KEY] = {
                'index': 'chunk',
                'key': self._clean_id(chunk_id)
            }

            chunk_node.metadata = metadata
            chunk_node.excluded_embed_metadata_keys = [INDEX_KEY, 'chunk']
            chunk_node.excluded_llm_metadata_keys = [INDEX_KEY, 'chunk']

            chunk_nodes.append(chunk_node)

        return chunk_nodes

    
