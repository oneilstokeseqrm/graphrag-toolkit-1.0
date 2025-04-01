# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict
from graphrag_toolkit.storage.constants import ALL_EMBEDDING_INDEXES
from graphrag_toolkit.storage.vector.vector_index import VectorIndex
from graphrag_toolkit.storage.vector.dummy_vector_index import DummyVectorIndex
from llama_index.core.bridge.pydantic import BaseModel, Field

class VectorStore(BaseModel):
    indexes:Dict[str, VectorIndex] = Field(description='Vector indexes')

    def __init__(self, indexes:List[VectorIndex]=[]):
        super().__init__(indexes={i.index_name:i for i in indexes})

    def get_index(self, index_name):
        if index_name not in ALL_EMBEDDING_INDEXES:
            raise ValueError(f'Invalid index name ({index_name}): must be one of {ALL_EMBEDDING_INDEXES}')
        if index_name not in self.indexes:
            return DummyVectorIndex(index_name=index_name)
        return self.indexes[index_name]
