# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import List, Optional

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector.vector_store import VectorStore

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

class SemanticGuidedBaseRetriever(BaseRetriever):

    def __init__(self, 
                vector_store:VectorStore,
                graph_store:GraphStore,
                filter_config:Optional[FilterConfig]=None,
                **kwargs):
        
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.filter_config = filter_config or FilterConfig()
        self.debug_results = kwargs.pop('debug_results', None) is not None

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError()