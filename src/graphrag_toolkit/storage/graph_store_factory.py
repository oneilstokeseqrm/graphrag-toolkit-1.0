# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union, Type, Dict

from graphrag_toolkit.storage.graph import GraphStore, GraphStoreFactoryMethod
from graphrag_toolkit.storage.graph.dummy_graph_store import DummyGraphStoreFactory
from graphrag_toolkit.storage.graph.neptune_graph_stores import NeptuneAnalyticsGraphStoreFactory, NeptuneDatabaseGraphStoreFactory

logger = logging.getLogger(__name__)

GraphStoreType = Union[str, GraphStore]
GraphStoreFactoryMethodType = Union[GraphStoreFactoryMethod, Type[GraphStoreFactoryMethod]]

_graph_store_factories:Dict[str, GraphStoreFactoryMethod] = { c.__name__ : c() for c in [NeptuneAnalyticsGraphStoreFactory, NeptuneDatabaseGraphStoreFactory, DummyGraphStoreFactory] }


class GraphStoreFactory():

    @staticmethod
    def register(factory_type:GraphStoreFactoryMethodType):
        if isinstance(factory_type, type):
            if not issubclass(factory_type, GraphStoreFactoryMethod):
                raise ValueError(f'Invalid factory_type argument: {factory_type.__name__} must inherit from GraphStoreFactoryMethod.')
            _graph_store_factories[factory_type.__name__] = factory_type()
        else:
            factory_type_name = type(factory_type).__name__
            if not isinstance(factory_type, GraphStoreFactoryMethod):
                raise ValueError(f'Invalid factory_type argument: {factory_type_name} must inherit from GraphStoreFactoryMethod.')
            _graph_store_factories[factory_type_name] = factory_type

    @staticmethod
    def for_graph_store(graph_info:GraphStoreType=None, **kwargs) -> GraphStore:

        if graph_info and isinstance(graph_info, GraphStore):
            return graph_info
        
        for factory in _graph_store_factories.values():
            graph_store = factory.try_create(graph_info, **kwargs)
            if graph_store:
                return graph_store
            
        raise ValueError(f'Unrecognized graph store info: {graph_info}. Check that the graph store connection info is formatted correctly, and that an appropriate graph store factory method is registered with GraphStoreFactory.')

