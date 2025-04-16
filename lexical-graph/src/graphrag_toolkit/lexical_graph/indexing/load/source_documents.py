# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List
from llama_index.core import Document

class SourceDocuments:
    def __init__(self, source_documents_fns: List[Callable[[], List[Document] ]]):    
        self.source_documents_fns = source_documents_fns
        
    def __iter__(self):
        for source_documents_fn in self.source_documents_fns:
            for source_documents in source_documents_fn(): 
                if isinstance(source_documents, list):              
                    for item in source_documents:                        
                        if isinstance(item, list):
                            for i in item:
                                yield i
                        else:
                            yield item
                else:
                    yield source_documents