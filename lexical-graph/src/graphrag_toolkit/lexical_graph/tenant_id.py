# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union
from llama_index.core.bridge.pydantic import BaseModel

class TenantId(BaseModel):

    value:Optional[str]=None

    def __init__(self, value:str=None):
        if value is not None:
            if len(value) > 10 or len(value) < 1 or not value.isalnum() or any(letter.isupper() for letter in value):
                raise ValueError(f"Invalid TenantId: '{value}'. TenantId must be between 1-10 lowercase letters and numbers.")
        super().__init__(value=value)
        
    def is_default_tenant(self):
        return self.value is None
    
    def format_label(self, label:str):
        if self.is_default_tenant():
            return f'`{label}`'
        return f'`{label}{self.value}__`'
    
    def format_index_name(self, index_name:str):
        if self.is_default_tenant():
            return index_name
        return f'{index_name}_{self.value}'
    
    def format_hashable(self, hashable:str):
        if self.is_default_tenant():
            return hashable
        else:
            return f'{self.value}::{hashable}'
        
    def format_id(self, prefix:str, id_value:str):
        if self.is_default_tenant():
            return f'{prefix}::{id_value}'
        else:
            return f'{prefix}:{self.value}:{id_value}'
        
    def rewrite_id(self, id_value:str):
        if self.is_default_tenant():
            return id_value
        else:
            id_parts = id_value.split(':')
            return f'{id_parts[0]}:{self.value}:{":".join(id_parts[2:])}'
        
DEFAULT_TENANT_ID = TenantId()

TenantIdType = Union[str, TenantId]

def to_tenant_id(tenant_id:Optional[TenantIdType]):
    if tenant_id is None:
        return DEFAULT_TENANT_ID
    if isinstance(tenant_id, TenantId):
        return tenant_id
    else:
        return TenantId(str(tenant_id))
    