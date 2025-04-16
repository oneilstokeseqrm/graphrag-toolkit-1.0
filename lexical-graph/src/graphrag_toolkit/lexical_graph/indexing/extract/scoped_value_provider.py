# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
from typing import Dict, List, Callable, Tuple

from llama_index.core.schema import BaseComponent, BaseNode
from llama_index.core.bridge.pydantic import Field

DEFAULT_SCOPE = '__ALL__'

logger = logging.getLogger(__name__)

def default_scope_fn(node):
    return DEFAULT_SCOPE

class ScopedValueStore(BaseComponent):

    @abc.abstractmethod
    def get_scoped_values(self, label:str, scope:str) -> List[str]:
        pass

    @abc.abstractmethod
    def save_scoped_values(self, label:str, scope:str, values:List[str]) -> None:
        pass    

class FixedScopedValueStore(ScopedValueStore):
    scoped_values:Dict[str,List[str]] = Field(default={})

    def get_scoped_values(self, label:str, scope:str) -> List[str]:
        return self.scoped_values.get(scope, [])

    def save_scoped_values(self, label:str, scope:str, values:List[str]) -> None:
        pass

class ScopedValueProvider(BaseComponent):

    label:str = Field(
        description='Scoped value label'
    )

    scope_func:Callable[[BaseNode], str] = Field(
        description='Function for determining scope given an input node'
    )

    scoped_value_store:ScopedValueStore = Field(
        description='Scoped value store'
    )

    @classmethod
    def class_name(cls) -> str:
        return 'ScopedValueProvider'

    def __init__(self, 
                 label:str,
                 scoped_value_store: ScopedValueStore,
                 scope_func:Callable[[BaseNode], str]=None, 
                 initial_scoped_values: Dict[str, List[str]]={}):
        
        for k,v in initial_scoped_values.items():
            scoped_value_store.save_scoped_values(label, k, v)
        
        super().__init__(
            label=label,
            scope_func=scope_func or default_scope_fn,
            scoped_value_store=scoped_value_store
        )

    def get_current_values(self, node:BaseNode) -> Tuple[str, List[str]]:
        scope = self.scope_func(node)
        current_values = self.scoped_value_store.get_scoped_values(self.label, scope)
        return (scope, current_values)
    
    def update_values(self, scope:str, old_values:List[str], new_values:List[str]):
        values = list(set(new_values).difference(set(old_values)))
        if values:
            logger.debug(f'Adding scoped values: [label: {self.label}, scope: {scope}, values: {values}]')
            self.scoped_value_store.save_scoped_values(self.label, scope, values)


class FixedScopedValueProvider(ScopedValueProvider):
    def __init__(self, scoped_values: Dict[str, List[str]]={}):
        super().__init__(
            label='__FIXED__',
            scoped_value_store=FixedScopedValueStore(scoped_values=scoped_values)
        )


