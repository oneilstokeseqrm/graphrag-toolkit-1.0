# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Iterable

from graphrag_toolkit.lexical_graph.indexing.model import SourceDocument

from llama_index.core.schema import BaseComponent

class SourceDocParser(BaseComponent):
     
    @abc.abstractmethod
    def _parse_source_docs(self, source_documents:Iterable[SourceDocument]) -> Iterable[SourceDocument]:
        pass

    def parse_source_docs(self, source_documents:Iterable[SourceDocument]) -> Iterable[SourceDocument]:
        return self._parse_source_docs(source_documents)