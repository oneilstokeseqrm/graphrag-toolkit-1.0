# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from tqdm import tqdm
from typing import List, Any, Union

from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.indexing.node_handler import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.build.vector_batch_client import VectorBatchClient
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY, ALL_EMBEDDING_INDEXES, DEFAULT_EMBEDDING_INDEXES

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

VectorStoreInfoType = Union[str, VectorStore]

class VectorIndexing(NodeHandler):

    @staticmethod
    def for_vector_store(vector_store_info:VectorStoreInfoType=None, index_names=DEFAULT_EMBEDDING_INDEXES, **kwargs):
        if isinstance(vector_store_info, VectorStore):
            return VectorIndexing(vector_store=vector_store_info)
        else:
            return VectorIndexing(vector_store=VectorStoreFactory.for_vector_store(vector_store_info, index_names, **kwargs))
    
    vector_store:VectorStore

    def accept(self, nodes: List[BaseNode], **kwargs: Any):

        batch_writes_enabled = kwargs.pop('batch_writes_enabled')
        batch_write_size = kwargs.pop('batch_write_size')

        logger.debug(f'Batch config: [batch_writes_enabled: {batch_writes_enabled}, batch_write_size: {batch_write_size}]')
        logger.debug(f'Vector indexing kwargs: {kwargs}')
        
        with VectorBatchClient(vector_store=self.vector_store, batch_writes_enabled=batch_writes_enabled, batch_write_size=batch_write_size) as batch_client:

            node_iterable = nodes if not self.show_progress else tqdm(nodes, desc=f'Building vector index [batch_writes_enabled: {batch_writes_enabled}, batch_write_size: {batch_write_size}]')

            for node in node_iterable:
                if [key for key in [INDEX_KEY] if key in node.metadata]:
                    try:
                        index_name = node.metadata[INDEX_KEY]['index']
                        if index_name in ALL_EMBEDDING_INDEXES:
                            index = batch_client.get_index(index_name)
                            index.add_embeddings([node])
                    except Exception as e:
                        logger.exception('An error occurred while indexing vectors')
                        raise e
                if batch_client.allow_yield(node):
                    yield node

            batch_nodes = batch_client.apply_batch_operations()
            for node in batch_nodes:
                yield node
        