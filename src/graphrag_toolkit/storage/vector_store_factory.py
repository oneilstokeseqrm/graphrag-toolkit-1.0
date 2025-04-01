# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Type
from graphrag_toolkit.storage.vector import VectorStore, VectorIndexFactoryMethod
from graphrag_toolkit.storage.vector.opensearch_vector_indexes import OpenSearchVectorIndexFactory
from graphrag_toolkit.storage.vector.neptune_vector_indexes import NeptuneAnalyticsVectorIndexFactory
from graphrag_toolkit.storage.vector.pg_vector_indexes import PGVectorIndexFactory
from graphrag_toolkit.storage.vector.dummy_vector_index import DummyVectorIndexFactory
from graphrag_toolkit.storage.constants import DEFAULT_EMBEDDING_INDEXES


VectorStoreType = Union[str, VectorStore]

_vector_index_factories = { OpenSearchVectorIndexFactory, PGVectorIndexFactory, NeptuneAnalyticsVectorIndexFactory, DummyVectorIndexFactory }

class VectorStoreFactory():

    @staticmethod
    def register_vector_index_factory_method(factory_type:Type[VectorIndexFactoryMethod]):
        if not isinstance(factory_type, type):
            raise ValueError(f'Invalid factory_type argument: must be a class not an instance.')
        if not issubclass(factory_type, VectorIndexFactoryMethod):
            raise ValueError(f'Invalid factory_type argument: {factory_type.__name__} must inherit from VectorIndexFactoryMethod.')
        _vector_index_factories.add(factory_type)

    @staticmethod
    def for_vector_store(vector_store_info:str=None, index_names=DEFAULT_EMBEDDING_INDEXES, **kwargs):
        if vector_store_info and isinstance(vector_store_info, VectorStore):
            return vector_store_info
        index_names = index_names if isinstance(index_names, list) else [index_names]

        for factory in _vector_index_factories:
            vector_indexes = factory().try_create(index_names, vector_store_info, **kwargs)
            if vector_indexes:
                return VectorStore(vector_indexes)
            
        raise ValueError(f'Unrecognized vector store info: {vector_store_info}. Check that the vector store connection info is formatted correctly, and that an appropriate vector index factory method is registered with VectorStoreFactory.')
    
    @staticmethod
    def for_composite(vector_store_list:List[VectorStore]):
        indexes = []
        for v in vector_store_list:
            indexes.extend(v.indexes.values())            
        return VectorStore(indexes)