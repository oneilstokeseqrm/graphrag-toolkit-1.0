# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from os.path import join
from typing import Any, List

from graphrag_toolkit.lexical_graph.indexing.node_handler import NodeHandler
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

from llama_index.core.schema import TransformComponent, BaseNode

SAVEPOINT_ROOT_DIR = 'save_points'

logger = logging.getLogger(__name__)

class DoNotCheckpoint:
    pass

class CheckpointFilter(TransformComponent, DoNotCheckpoint):
    
    checkpoint_name:str
    checkpoint_dir:str
    inner:TransformComponent
        
    def checkpoint_does_not_exist(self, node_id):
        node_checkpoint_path = join(self.checkpoint_dir, node_id)
        if os.path.exists(node_checkpoint_path):
            logger.debug(f'Ignoring node because checkpoint already exists [node_id: {node_id}, checkpoint: {self.checkpoint_name}, component: {type(self.inner).__name__}]')
            return False
        else:
            logger.debug(f'Including node [node_id: {node_id}, checkpoint: {self.checkpoint_name}, component: {type(self.inner).__name__}]')
            return True
        
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        filtered_nodes = [node for node in nodes if self.checkpoint_does_not_exist(node.id_)]
        return self.inner.__call__(filtered_nodes, **kwargs)

    
class CheckpointWriter(NodeHandler):

    checkpoint_name:str
    checkpoint_dir:str
    inner:NodeHandler

    def touch(self, path):
        with open(path, 'a'):
            os.utime(path, None)
    
    def accept(self, nodes: List[BaseNode], **kwargs: Any):
        
        for node in self.inner.accept(nodes, **kwargs):
            node_id = node.node_id
            if [key for key in [INDEX_KEY] if key in node.metadata]:
                logger.debug(f'Non-checkpointable node [checkpoint: {self.checkpoint_name}, node_id: {node_id}, component: {type(self.inner).__name__}]') 
            else:
                logger.debug(f'Checkpointable node [checkpoint: {self.checkpoint_name}, node_id: {node_id}, component: {type(self.inner).__name__}]') 
                node_checkpoint_path = join(self.checkpoint_dir, node_id)
                self.touch(node_checkpoint_path)
            yield node

class Checkpoint():

    
    def __init__(self, checkpoint_name, output_dir='output', enabled=True):
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = self.prepare_output_directories(checkpoint_name, output_dir)
        self.enabled = enabled

    def add_filter(self, o):
        if self.enabled and isinstance(o, TransformComponent) and not isinstance(o, DoNotCheckpoint):
            logger.debug(f'Wrapping with checkpoint filter [checkpoint: {self.checkpoint_name}, component: {type(o).__name__}]')
            return CheckpointFilter(inner=o, checkpoint_dir=self.checkpoint_dir, checkpoint_name=self.checkpoint_name)
        else:
            logger.debug(f'Not wrapping with checkpoint filter [checkpoint: {self.checkpoint_name}, component: {type(o).__name__}]')
            return o
        
    def add_writer(self, o):
        if self.enabled and isinstance(o, NodeHandler):
            logger.debug(f'Wrapping with checkpoint writer [checkpoint: {self.checkpoint_name}, component: {type(o).__name__}]')
            return CheckpointWriter(inner=o, checkpoint_dir=self.checkpoint_dir, checkpoint_name=self.checkpoint_name)
        else:
            logger.debug(f'Not wrapping with checkpoint writer [checkpoint: {self.checkpoint_name}, component: {type(o).__name__}]')
            return o

    def prepare_output_directories(self, checkpoint_name, output_dir):
        
        checkpoint_dir = join(output_dir, SAVEPOINT_ROOT_DIR, checkpoint_name)
        
        logger.debug(f'Preparing checkpoint directory [checkpoint: {checkpoint_name}, checkpoint_dir: {checkpoint_dir}]')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
   
        return checkpoint_dir

    
