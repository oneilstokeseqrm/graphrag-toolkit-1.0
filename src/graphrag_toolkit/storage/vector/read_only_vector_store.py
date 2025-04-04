# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
from graphrag_toolkit import TenantId
from graphrag_toolkit.storage.vector import VectorStore, VectorIndex


class ReadOnlyVectorStore(VectorStore):

    @classmethod
    def wrap(cls, vector_store:VectorStore):
        if isinstance(vector_store, ReadOnlyVectorStore):
            return vector_store
        return ReadOnlyVectorStore(inner=vector_store)

    inner:VectorStore

    def get_index(self, index_name):
        index = self.inner.get_index(index_name=index_name)
        index.writeable = False
        return index
    
    def all_indexes(self) -> List[VectorIndex]:
        return [self.get_index(i) for i in self.inner.indexes.keys()]
        