# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Dict, List
from llama_index.core.schema import BaseNode, BaseComponent

from graphrag_toolkit.indexing import IdGenerator
from graphrag_toolkit.indexing.build.build_filter import BuildFilter
from graphrag_toolkit.indexing.constants import DEFAULT_CLASSIFICATION

class NodeBuilder(BaseComponent):

    id_generator:IdGenerator

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def metadata_keys(cls) -> List[str]:
        pass

    @abc.abstractmethod
    def build_nodes(self, nodes:List[BaseNode], filter:BuildFilter) -> List[BaseNode]:
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
