# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import abc
from typing import List

from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

class KeywordProviderBase():
    
    def __init__(self, args:ProcessorArgs):    
        self.args = args

    @abc.abstractmethod
    def get_keywords(self, query_bundle:QueryBundle) -> List[str]:
        ...