# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict, List, Optional, Protocol, Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class TextExtractorFunction(Protocol):
    def __call__(self, data:Dict[str,Any]) -> str:
        pass

class MetadataExtractorFunction(Protocol):
    def __call__(self, data:Dict[str,Any]) -> Dict[str,Any]:
        pass

class JSONArrayReader(BaseReader):

    def __init__(self, ensure_ascii:bool=False, text_fn:Optional[TextExtractorFunction]=None, metadata_fn=Optional[MetadataExtractorFunction]):
        super().__init__()
        self.ensure_ascii = ensure_ascii
        self.text_fn = text_fn
        self.metadata_fn = metadata_fn
        
    def _get_metadata(self, data:Dict, extra_info:Dict):
        
        metadata = {}
        
        if extra_info:
            metadata.update(extra_info)
        if self.metadata_fn:
            metadata.update(self.metadata_fn(data))
        return metadata

    def load_data(self, input_file: str, extra_info: Optional[Dict] = {}) -> List[Document]:
        
        with open(input_file, encoding='utf-8') as f:
            json_data = json.load(f)
            
            if not isinstance(json_data, list):
                json_data = [json_data]

            documents = []

            for data in json_data:
                if self.text_fn:
                    text = self.text_fn(data)
                    metadata = self._get_metadata(data, extra_info)
                    documents.append(Document(text=text, metadata=metadata))
                else:
                    json_output = json.dumps(data, ensure_ascii=self.ensure_ascii)
                    documents.append(Document(text=json_output, metadata=self._get_metadata(data, extra_info)))
            
            return documents