# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any, Sequence

from graphrag_toolkit.lexical_graph.indexing.build.checkpoint import DoNotCheckpoint

from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.node_parser.node_utils import build_nodes_from_splits

logger = logging.getLogger(__name__)

class DocsToNodes(NodeParser, DoNotCheckpoint):
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
    
        def to_node(node):
            if isinstance(node, Document):
                return build_nodes_from_splits([node.text], node)[0]
            else:
                return node
    
        return [to_node(n) for n in nodes]