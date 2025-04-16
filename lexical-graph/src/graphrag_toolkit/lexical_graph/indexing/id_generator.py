# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from typing import Optional

from graphrag_toolkit.lexical_graph import TenantId

from llama_index.core.bridge.pydantic import BaseModel, Field

class IdGenerator(BaseModel):
    tenant_id:TenantId = Field(default_factory=lambda: TenantId())

    def _get_hash(self, s):
        return hashlib.md5(s.encode('utf-8')).digest().hex()

    def create_source_id(self, text:str, metadata_str:str):
        return f"aws::{self._get_hash(text)[:8]}:{self._get_hash(metadata_str)[:4]}"
        
    def create_chunk_id(self, source_id:str, text:str, metadata_str:str):
        return f'{source_id}:{self._get_hash(text + metadata_str)[:8]}'
    
    def rewrite_id_for_tenant(self, id_value:str):
        return self.tenant_id.rewrite_id(id_value)

    def create_node_id(self, node_type:str, v1:str, v2:Optional[str]=None) -> str:
        if v2:
            return self._get_hash(self.tenant_id.format_hashable(f"{node_type.lower()}::{v1.lower().replace(' ', '_')}::{v2.lower().replace(' ', '_')}"))
        else:
            return self._get_hash(self.tenant_id.format_hashable(f"{node_type.lower()}::{v1.lower().replace(' ', '_')}"))