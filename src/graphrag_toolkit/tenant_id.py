# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from llama_index.core.schema import BaseNode

class TenantId(BaseNode):

    value:str=None

    def __init__(self, value:str=None):
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