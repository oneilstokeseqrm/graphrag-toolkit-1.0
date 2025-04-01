# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import List

from graphrag_toolkit.storage.graph import GraphStore
from graphrag_toolkit.storage.vector_store import VectorStore

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

class SemanticGuidedBaseRetriever(BaseRetriever):

    def __init__(self, 
                vector_store:VectorStore,
                graph_store:GraphStore,
                **kwargs):
        
        self.graph_store = graph_store
        self.vector_store = vector_store

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError()