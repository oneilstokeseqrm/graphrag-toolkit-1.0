# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic

from llama_index.core.schema import QueryBundle

class ClearScores(ProcessorBase):
    def __init__(self, args:ProcessorArgs):
        super().__init__(args)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        
        def clear_score(index:int, search_result:SearchResult):
            search_result.score = None
            return search_result
        
        return self._apply_to_search_results(search_results, clear_score)
