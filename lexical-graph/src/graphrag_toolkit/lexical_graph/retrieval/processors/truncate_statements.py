# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic
from llama_index.core.schema import QueryBundle

class TruncateStatements(ProcessorBase):
    def __init__(self, args:ProcessorArgs):
        super().__init__(args)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        def truncate_statements(topic:Topic):
            topic.statements = topic.statements[:self.args.max_statements_per_topic]
            return topic
        
        def truncate_search_result(index:int, search_result:SearchResult):
            return self._apply_to_topics(search_result, truncate_statements)
        
        return self._apply_to_search_results(search_results, truncate_search_result)


