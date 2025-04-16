# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from string import Template
from typing import Optional, List, Union, Dict, Any, Callable

from graphrag_toolkit.lexical_graph.retrieval.model import SearchResult

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

SourceInfoTemplateType = Union[str, Template]
SourceInfoAccessorType = Union[str, List[str], Template, Callable[[Dict[str, Any]], str]]

def source_info_template(template:SourceInfoTemplateType) -> Callable[[Dict[str, Any]], str]:
    t = template if isinstance(template, Template) else Template(template)
    def source_info_template_fn(source_properties:Dict[str, Any]) -> str:
        return t.safe_substitute(source_properties)
    return source_info_template_fn

def source_info_keys(keys:List[str]) -> Callable[[Dict[str, Any]], str]:
    def source_info_keys_fn(source_properties:Dict[str, Any]) -> str:
        for key in keys:
            if key in source_properties:
                return source_properties[key]
        return None
    return source_info_keys_fn

class EnrichSourceDetails(BaseNodePostprocessor):

    source_info_accessor:SourceInfoAccessorType=None

    @classmethod
    def class_name(cls) -> str:
        return 'EnrichSourceDetails'
    
    def _get_source_info(self, source_metadata, source) -> str:
        
        accessor = self.source_info_accessor
        
        if not accessor:
            return source

        if isinstance(accessor, str):
            fn = source_info_template(accessor) if '$' in accessor else source_info_keys([accessor])
        if isinstance(accessor, list):
            fn = source_info_keys(accessor)
        if isinstance(accessor, Template):
            fn = source_info_template(accessor)
        if isinstance(accessor, Callable):
            fn = accessor

        source_info = fn(source_metadata)

        return source_info or source

    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        
        for node in nodes:
            search_result = SearchResult.model_validate_json(node.node.text)
            source_metadata = node.metadata.get('source', {}).get('metadata', {})
            if source_metadata:
                source_info = self._get_source_info(source_metadata, search_result.source.sourceId)
                search_result.source = str(source_info)
                node.node.text = search_result.model_dump_json(exclude_none=True, exclude_defaults=True, indent=2)

        return nodes

