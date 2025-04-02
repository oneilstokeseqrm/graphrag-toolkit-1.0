# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.indexing.model import Topic
from graphrag_toolkit.storage.graph import GraphStore
from graphrag_toolkit.indexing.build.graph_builder import GraphBuilder

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class TopicGraphBuilder(GraphBuilder):
    
    @classmethod
    def index_key(cls) -> str:
        return 'topic'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
            
        topic_metadata = node.metadata.get('topic', {})

        if topic_metadata:

            topic = Topic.model_validate(topic_metadata)
        
            logger.debug(f'Inserting topic [topic_id: {topic.topicId}]')

            statements = [
                '// insert topics',
                'UNWIND $params AS params'
            ]
            

            chunk_ids =  [ {'chunk_id': chunkId} for chunkId in topic.chunkIds]

            statements.extend([
                f'MERGE (topic:`__Topic__`{{{graph_client.node_id("topicId")}: params.topic_id}})',
                'ON CREATE SET topic.value=params.title',
                'ON MATCH SET topic.value=params.title',
                'WITH topic, params',
                'UNWIND params.chunk_ids as chunkIds',
                f'MERGE (chunk:`__Chunk__`{{{graph_client.node_id("chunkId")}: chunkIds.chunk_id}})',
                'MERGE (topic)-[:`__MENTIONED_IN__`]->(chunk)'
            ])

            properties = {
                'topic_id': topic.topicId,
                'title': topic.value,
                'chunk_ids': chunk_ids
            }

            query = '\n'.join(statements)

            graph_client.execute_query_with_retry(query, self._to_params(properties))

        else:
            logger.warning(f'topic_id missing from topic node [node_id: {node.node_id}]') 
