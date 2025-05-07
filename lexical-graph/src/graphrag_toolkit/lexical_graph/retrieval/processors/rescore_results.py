# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import statistics
from typing import List, Dict

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult

from llama_index.core.schema import QueryBundle

class RescoreResults(ProcessorBase):
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        super().__init__(args, filter_config)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:

        def rescore_search_result(index:int, search_result:SearchResult):
            topic_scores = [
                max([s.score for s in topic.statements])
                for topic in search_result.topics
            ]
            
            search_result.score = statistics.mean(topic_scores)
            
            return search_result
        
        return self._apply_to_search_results(search_results, rescore_search_result)
        


