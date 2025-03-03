# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from typing import Optional

from llama_index.core.bridge.pydantic import BaseModel

class IdGenerator(BaseModel):
    tenant_id:Optional[str]=None

    def _get_hash(self, s):
        return hashlib.md5(s.encode('utf-8')).digest().hex()

    def create_source_id(self, text:str, metadata_str:str):
        return f"aws:{self.tenant_id if self.tenant_id else ''}:{self._get_hash(text)[:8]}:{self._get_hash(metadata_str)[:4]}"
    
    def create_chunk_id(self, source_id:str, text:str, metadata_str:str):
        return f'{source_id}:{self._get_hash(text + metadata_str)[:8]}'
    
    def create_node_id(self, node_type:str, v1:str, v2:Optional[str]=None) -> str:
        tenant_id_prefix = f'{self.tenant_id}::' if self.tenant_id else ''
        if v2:
            return self._get_hash(f"{tenant_id_prefix}{node_type.lower()}::{v1.lower().replace(' ', '_')}::{v2.lower().replace(' ', '_')}")
        else:
            return self._get_hash(f"{tenant_id_prefix}{node_type.lower()}::{v1.lower().replace(' ', '_')}")