# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic

from llama_index.core.schema import QueryBundle

class ClearChunks(ProcessorBase):
    def __init__(self, args:ProcessorArgs):
        super().__init__(args)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        def clear_chunks(topic:Topic):
            topic.chunks.clear()
            return topic

        def clear_search_result_chunks(index:int, search_result:SearchResult):
            return self._apply_to_topics(search_result, clear_chunks)
        
        return self._apply_to_search_results(search_results, clear_search_result_chunks)


