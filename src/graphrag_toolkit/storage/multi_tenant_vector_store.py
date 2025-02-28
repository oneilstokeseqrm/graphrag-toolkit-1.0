# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List
from graphrag_toolkit.storage.vector_store import VectorStore
from graphrag_toolkit.storage.vector_index import VectorIndex

from llama_index.core.bridge.pydantic import Field

class MultiTenantVectorStore(VectorStore):

    @classmethod
    def wrap(cls, vector_store:VectorStore, graph_name:Optional[str]=None):
        if not graph_name:
            return vector_store
        if isinstance(vector_store, MultiTenantVectorStore):
            return vector_store
        return MultiTenantVectorStore(inner=vector_store, graph_name=graph_name)

    inner:VectorStore
    graph_name:Optional[str]=None

    def get_index(self, index_name):
        index = self.inner.get_index(index_name=index_name)
        index.graph_name = self.graph_name
        return index
    
    def all_indexes(self) -> List[VectorIndex]:
        return [self.get_index(i) for i in self.inner.indexes.keys()]
        