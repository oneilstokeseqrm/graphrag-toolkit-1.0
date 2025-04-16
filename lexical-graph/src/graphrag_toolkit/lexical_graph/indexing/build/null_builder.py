# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any

from graphrag_toolkit.lexical_graph.indexing import NodeHandler

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class NullBuilder(NodeHandler):

    def accept(self, nodes: List[BaseNode], **kwargs: Any):

        for node in nodes:
            logger.debug(f'Accepted node [node_id: {node.node_id}]')         
            yield node