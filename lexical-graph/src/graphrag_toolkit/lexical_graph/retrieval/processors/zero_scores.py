# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic

from llama_index.core.schema import QueryBundle

class ZeroScores(ProcessorBase):
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        super().__init__(args, filter_config)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        def zero_statement_scores(topic:Topic):
            for s in topic.statements:
                s.score = 0.0
            return topic

        def zero_search_result_scores(index:int, search_result:SearchResult):
            search_result.score = 0.0
            return self._apply_to_topics(search_result, zero_statement_scores)
        
        return self._apply_to_search_results(search_results, zero_search_result_scores)


