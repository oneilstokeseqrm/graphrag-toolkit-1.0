# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Optional, List

from graphrag_toolkit.lexical_graph.storage.constants import ALL_EMBEDDING_INDEXES
from graphrag_toolkit.lexical_graph.storage.vector.vector_index import VectorIndex
from graphrag_toolkit.lexical_graph.storage.vector.dummy_vector_index import DummyVectorIndex

from llama_index.core.bridge.pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class VectorStore(BaseModel):
    indexes:Optional[Dict[str, VectorIndex]] = Field(description='Vector indexes', default_factory=dict)

    def get_index(self, index_name):
        if index_name not in ALL_EMBEDDING_INDEXES:
            raise ValueError(f'Invalid index name ({index_name}): must be one of {ALL_EMBEDDING_INDEXES}')
        if index_name not in self.indexes:
            logger.debug(f"Returning dummy index for '{index_name}'")
            return DummyVectorIndex(index_name=index_name)
        return self.indexes[index_name]

    def all_indexes(self) -> List[VectorIndex]:
        return list(self.indexes.values())

