# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import List, Any, Generator
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TransformComponent
from llama_index.core.bridge.pydantic import Field

class NodeHandler(TransformComponent):

    show_progress: bool = Field(default=True, description='Whether to show progress.')

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        return [n for n in self.accept(nodes, **kwargs)]
    
    @abc.abstractmethod
    def accept(self, nodes: List[BaseNode], **kwargs: Any) -> Generator[BaseNode, None, None]:
        raise NotImplementedError()