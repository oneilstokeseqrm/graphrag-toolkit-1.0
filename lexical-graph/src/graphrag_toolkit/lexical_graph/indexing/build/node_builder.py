# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import List
from llama_index.core.schema import BaseNode, BaseComponent

from graphrag_toolkit.lexical_graph.metadata import SourceMetadataFormatter
from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.build.build_filters import BuildFilters
from graphrag_toolkit.lexical_graph.indexing.constants import DEFAULT_CLASSIFICATION

class NodeBuilder(BaseComponent):
    
    id_generator:IdGenerator
    build_filters:BuildFilters
    source_metadata_formatter:SourceMetadataFormatter

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def metadata_keys(cls) -> List[str]:
        pass

    @abc.abstractmethod
    def build_nodes(self, nodes:List[BaseNode]) -> List[BaseNode]:
        pass
    
    def _clean_id(self, s):
        return ''.join(c for c in s if c.isalnum())
        
    def _format_classification(self, classification):
        if not classification or classification == DEFAULT_CLASSIFICATION:
            return ''
        else:
            return f' ({classification})'
    
    def _format_fact(self, s, sc, p, o, oc):
        return f'{s} {p} {o}'
