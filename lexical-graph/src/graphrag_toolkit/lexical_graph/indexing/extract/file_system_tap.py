# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import json
from typing import List, Iterable
from os.path import join

from graphrag_toolkit.lexical_graph.indexing.extract.pipeline_decorator import PipelineDecorator
from graphrag_toolkit.lexical_graph.indexing.model import SourceDocument

from llama_index.core.schema import Document, BaseNode

logger = logging.getLogger(__name__)
             
class FileSystemTap(PipelineDecorator):
    
    def __init__(self, subdirectory_name, clean=True, output_dir='output'):

        (raw_sources_dir, chunks_dir, sources_dir) = self._prepare_output_directories(output_dir, subdirectory_name, clean)

        self.raw_sources_dir = raw_sources_dir
        self.chunks_dir = chunks_dir
        self.sources_dir = sources_dir

    def handle_input_docs(self, docs:Iterable[SourceDocument]) -> Iterable[SourceDocument]:
        for doc in docs:
            if doc.refNode and isinstance(doc.refNode, Document):
                ref_node = doc.refNode
                raw_source_output_path = join(self.raw_sources_dir, ref_node.doc_id)
                source_output_path = join(self.sources_dir, f'{ref_node.doc_id}.json')
                with open(raw_source_output_path, 'w') as f:
                    f.write(ref_node.text)
                with open(source_output_path, 'w') as f:
                    f.write(ref_node.to_json())
        return docs
    
    def handle_output_doc(self, doc:SourceDocument) -> SourceDocument:
        for node in doc.nodes:
            chunk_output_path = join(self.chunks_dir, f'{node.node_id}.json')
            with open(chunk_output_path, 'w') as f:
                json.dump(node.to_dict(), f, indent=4)
        return doc    
        
    def _prepare_output_directories(self, output_dir, subdirectory_name, clean):
        
        raw_sources_dir = join(output_dir, 'extracted', subdirectory_name, 'raw')
        chunks_dir = join(output_dir, 'extracted', subdirectory_name, 'chunks')
        sources_dir = join(output_dir, 'extracted', subdirectory_name, 'sources')

        logger.info(f'Preparing output directories [subdirectory_name: {subdirectory_name}, raw_sources_dir: {raw_sources_dir}, chunks_dir: {chunks_dir}, sources_dir: {sources_dir}, clean: {clean}]')
        
        if clean:
            if os.path.exists(raw_sources_dir):
                shutil.rmtree(raw_sources_dir)
            if os.path.exists(chunks_dir):
                shutil.rmtree(chunks_dir)
            if os.path.exists(sources_dir):
                shutil.rmtree(sources_dir)
        
        if not os.path.exists(raw_sources_dir):
            os.makedirs(raw_sources_dir)
        if not os.path.exists(chunks_dir):
            os.makedirs(chunks_dir)
        if not os.path.exists(sources_dir):
            os.makedirs(sources_dir)
  
        return (raw_sources_dir, chunks_dir, sources_dir)
    



    