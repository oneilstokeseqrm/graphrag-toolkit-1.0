# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import abc

from typing import Sequence, Any, List, Dict
from llama_index.core.schema import QueryBundle, BaseNode
from llama_index.core.bridge.pydantic import BaseModel, Field, field_validator

from graphrag_toolkit import EmbeddingType, TenantId
from graphrag_toolkit.storage.constants import ALL_EMBEDDING_INDEXES

logger = logging.getLogger(__name__)

def to_embedded_query(query_bundle:QueryBundle, embed_model:EmbeddingType) -> QueryBundle:
    if query_bundle.embedding:
        return query_bundle
    
    query_bundle.embedding = (
        embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )
    ) 
    return query_bundle   

class VectorIndex(BaseModel):
    index_name: str
    tenant_id:TenantId = Field(default_factory=lambda: TenantId())

    @field_validator('index_name')
    def validate_option(cls, v):
        if v not in ALL_EMBEDDING_INDEXES:
            raise ValueError(f'Invalid index_name: must be one of {ALL_EMBEDDING_INDEXES}')
        return v
    
    def underlying_index_name(self) -> str:
        if self.tenant_id.is_default_tenant():
            return self.index_name
        else:
            return self.tenant_id.format_index_name(self.index_name)
    
    @abc.abstractmethod
    def add_embeddings(self, nodes:Sequence[BaseNode]) -> Sequence[BaseNode]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def top_k(self, query_bundle:QueryBundle, top_k:int=5) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_embeddings(self, ids:List[str]=[]) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError
