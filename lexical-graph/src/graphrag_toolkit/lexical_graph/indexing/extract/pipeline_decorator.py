# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import six
from typing import Iterable

from graphrag_toolkit.lexical_graph.indexing.model import SourceDocument

@six.add_metaclass(abc.ABCMeta)
class PipelineDecorator():

    @abc.abstractmethod
    def handle_input_docs(self, docs:Iterable[SourceDocument]) -> Iterable[SourceDocument]:
        pass

    @abc.abstractmethod
    def handle_output_doc(self, doc: SourceDocument) -> SourceDocument:
        pass

