# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult

from llama_index.core.schema import QueryBundle

class DisaggregateResults(ProcessorBase):
    def __init__(self, args:ProcessorArgs):
        super().__init__(args)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        disaggregated_results = []

        for search_result in search_results.results:
            for topic in search_result.topics:
                score = max([s.score for s in topic.statements])
                disaggregated_results.append(SearchResult(topics=[topic], source=search_result.source, score=score))
                
        search_results = search_results.with_new_results(results=disaggregated_results)
        
        return search_results


