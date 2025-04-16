# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
from datetime import datetime
from os.path import join
from typing import List, Any, Generator, Optional, Dict

from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY, TOPICS_KEY
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY 

from llama_index.core.schema import TextNode, BaseNode

logger = logging.getLogger(__name__)

class FileBasedChunks(NodeHandler):

    chunks_directory:str
    collection_id:str

    metadata_keys:Optional[List[str]]
    
    def __init__(self, 
                 chunks_directory:str, 
                 collection_id:Optional[str]=None,
                 metadata_keys:Optional[List[str]]=None):
        super().__init__(
            chunks_directory=chunks_directory,
            collection_id=collection_id or datetime.now().strftime('%Y%m%d-%H%M%S'),
            metadata_keys=metadata_keys
        )
        self._prepare_directory()

    def _prepare_directory(self):
        directory_path = join(self.chunks_directory, self.collection_id)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def chunks(self):
        return self
    
    def _filter_metadata(self, node:TextNode) -> TextNode:

        def filter(metadata:Dict):
            keys_to_delete = []
            for key in metadata.keys():
                if key not in [PROPOSITIONS_KEY, TOPICS_KEY, INDEX_KEY]:
                    if self.metadata_keys is not None and key not in self.metadata_keys:
                        keys_to_delete.append(key)
            for key in keys_to_delete:
                del metadata[key]

        filter(node.metadata)

        for _, relationship_info in node.relationships.items():
            if relationship_info.metadata:
                filter(relationship_info.metadata)

        return node

    def __iter__(self):
        directory_path = join(self.chunks_directory, self.collection_id)
        logger.debug(f'Reading chunks from directory: {directory_path}')
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path) as f:
                    yield self._filter_metadata(TextNode.from_json(f.read()))

    def accept(self, nodes: List[BaseNode], **kwargs: Any) -> Generator[BaseNode, None, None]:
        for n in nodes:
            if not [key for key in [INDEX_KEY] if key in n.metadata]:
                chunk_output_path = join(self.chunks_directory, self.collection_id, f'{n.node_id}.json')
                logger.debug(f'Writing chunk to file: {chunk_output_path}')
                with open(chunk_output_path, 'w') as f:
                    json.dump(n.to_dict(), f, indent=4)
            yield n