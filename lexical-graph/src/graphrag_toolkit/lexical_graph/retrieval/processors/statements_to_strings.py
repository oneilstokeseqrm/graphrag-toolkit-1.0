# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic

from llama_index.core.schema import QueryBundle

class StatementsToStrings(ProcessorBase):
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        super().__init__(args, filter_config)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        def get_statement_string(s):
            return s.statement_str if self.args.include_facts else s.statement
        
        def statements_to_strings(topic:Topic):
            topic.statements = [
                get_statement_string(statement)
                for statement in topic.statements
            ]
            return topic
    
        def search_result_statements_to_strings(index:int, search_result:SearchResult):
            return self._apply_to_topics(search_result, statements_to_strings)
        
        return self._apply_to_search_results(search_results, search_result_statements_to_strings)


