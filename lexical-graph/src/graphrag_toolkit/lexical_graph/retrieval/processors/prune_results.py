# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult

from llama_index.core.schema import QueryBundle

class PruneResults(ProcessorBase):
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        super().__init__(args, filter_config)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        def prune_search_result(index:int, search_result:SearchResult):
            return search_result if search_result.score >= self.args.results_pruning_threshold else None

        return self._apply_to_search_results(search_results, prune_search_result)


