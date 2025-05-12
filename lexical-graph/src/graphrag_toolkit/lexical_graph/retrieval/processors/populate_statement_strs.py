# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic

from llama_index.core.schema import QueryBundle

class PopulateStatementStrs(ProcessorBase):
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        super().__init__(args, filter_config)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        
        def populate_statement_strs(topic:Topic):
            for statement in topic.statements:
                statement_details = []
                if statement.facts:
                    statement_details.extend(statement.facts)
                if statement.details:
                    statement_details.extend(statement.details.split('\n'))
                statement.statement_str = (
                    f'{statement.statement} (details: {", ".join(statement_details)})' 
                    if statement_details 
                    else statement.statement
                )
            return topic

        def populate_search_result_statement_strs(index:int, search_result:SearchResult):
            return self._apply_to_topics(search_result, populate_statement_strs)
        
        return self._apply_to_search_results(search_results, populate_search_result_statement_strs)


