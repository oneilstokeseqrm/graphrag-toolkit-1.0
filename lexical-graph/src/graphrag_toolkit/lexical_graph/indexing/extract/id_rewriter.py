# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any, List, Sequence, Optional, Iterable

from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.build.checkpoint import DoNotCheckpoint
from graphrag_toolkit.lexical_graph.indexing.model import SourceDocument

from llama_index.core.schema import BaseNode, Document
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import NodeRelationship

class IdRewriter(NodeParser, DoNotCheckpoint):
    inner:Optional[NodeParser]=None
    id_generator:IdGenerator
    
    def _get_properties_str(self, properties, default):
        if properties:
            return ';'.join(sorted([f'{k}:{v}' for k,v in properties.items()]))
        else:
            return default
    
    def _new_doc_id(self, node):
        
        metadata_str = self._get_properties_str(node.metadata, node.doc_id)  
        return self.id_generator.create_source_id(str(node.text), metadata_str)     
        
    def _new_node_id(self, node):
        
        source_info = node.relationships.get(NodeRelationship.SOURCE, None)
        source_id = source_info.node_id if source_info else f'aws:{uuid.uuid4().hex}' 
        metadata_str = self._get_properties_str(node.metadata, node.node_id) 

        return self.id_generator.create_chunk_id(source_id, str(node.text), metadata_str)
        
        
    def _new_id(self, node):
        
        if node.id_.startswith('aws:'):
            return node.id_
        elif isinstance(node, Document):
            return self._new_doc_id(node)
        else:
            return self._new_node_id(node)
    
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
    
        id_mappings = {}
        
        for n in nodes:
            n.id_ = self._new_id(n)
            id_mappings[n.id_] = n.id_
                      
        if not self.inner:
            return nodes
            
        results = self.inner(nodes, **kwargs)
        
        for n in results:
            id_mappings[n.id_] = self._new_id(n)
        
        def update_ids(n):
            n.id_ = id_mappings[n.id_]
            for r in n.relationships.values():
                r.node_id = id_mappings.get(r.node_id, r.node_id)
            return n
            
        return [
            update_ids(n) 
            for n in results
        ]
    
    def handle_source_docs(self, source_documents:Iterable[SourceDocument]) -> List[SourceDocument]:
        for source_document in source_documents:
            if source_document.refNode:
                source_document.refNode = self._parse_nodes([source_document.refNode])[0]
            source_document.nodes = self._parse_nodes(source_document.nodes)
        return source_documents
    