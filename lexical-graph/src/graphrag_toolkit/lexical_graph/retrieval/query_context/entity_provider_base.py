# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
from typing import List, Optional

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs

class EntityProviderBase():
    
    def __init__(self, graph_store:GraphStore, args:ProcessorArgs, filter_config:Optional[FilterConfig]=None):
        self.graph_store = graph_store
        self.args = args
        self.filter_config = filter_config

    @abc.abstractmethod                 
    def get_entities(self, keywords:List[str])  -> List[ScoredEntity]:
        ...

        