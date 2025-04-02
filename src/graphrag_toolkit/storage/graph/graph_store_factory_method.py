# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc

from graphrag_toolkit.storage.graph.graph_store import GraphStore

class GraphStoreFactoryMethod():
    @abc.abstractmethod
    def try_create(self, graph_info:str, **kwargs) -> GraphStore:
        raise NotImplementedError
