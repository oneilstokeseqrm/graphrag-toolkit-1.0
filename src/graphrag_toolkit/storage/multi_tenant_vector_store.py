# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
from graphrag_toolkit import TenantId
from graphrag_toolkit.storage.vector import VectorStore, VectorIndex


class MultiTenantVectorStore(VectorStore):

    @classmethod
    def wrap(cls, vector_store:VectorStore, tenant_id:TenantId):
        if tenant_id.is_default_tenant():
            return vector_store
        if isinstance(vector_store, MultiTenantVectorStore):
            return vector_store
        return MultiTenantVectorStore(inner=vector_store, tenant_id=tenant_id)

    inner:VectorStore
    tenant_id:TenantId

    def get_index(self, index_name):
        index = self.inner.get_index(index_name=index_name)
        index.tenant_id = self.tenant_id
        return index
    
    def all_indexes(self) -> List[VectorIndex]:
        return [self.get_index(i) for i in self.inner.indexes.keys()]
        