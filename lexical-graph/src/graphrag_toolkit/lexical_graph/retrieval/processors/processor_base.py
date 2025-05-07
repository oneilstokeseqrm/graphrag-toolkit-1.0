# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import abc
from typing import Callable

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

class ProcessorBase(object):

    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        self.args = args
        self.filter_config = filter_config

    def _log_results(self, retriever_name:str, title:str, search_results:SearchResultCollection):
        processor_name = type(self).__name__
        if processor_name in self.args.debug_results and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'''Intermediate results [{retriever_name}.{processor_name}] {title}: {search_results.model_dump_json(
                indent=2, 
                exclude_unset=True, 
                exclude_defaults=True, 
                exclude_none=True, 
                warnings=False)
            }''')

    def _apply_to_search_results(self, 
                                 search_results:SearchResultCollection, 
                                 search_result_handler:Callable[[int, SearchResult], SearchResult],
                                 **kwargs):
        
        surviving_search_results = []

        for i, search_result in enumerate(search_results.results):
            return_result = search_result_handler(i, search_result, **kwargs)
            if return_result and return_result.topics or return_result.statements:
                surviving_search_results.append(return_result)

        return search_results.with_new_results(results=surviving_search_results)
    
    def _apply_to_topics(self, 
                         search_result:SearchResult, 
                         topic_handler:Callable[[Topic], Topic], 
                         **kwargs):

        surviving_topics = []

        for topic in search_result.topics:

            return_topic = topic_handler(topic, **kwargs)

            if return_topic and return_topic.statements:
                surviving_topics.append(return_topic)

        search_result.topics = surviving_topics

        return search_result
    
    def _format_statement_context(self, source_str:str, topic_str:str, statement_str:str):
            return f'{topic_str}: {statement_str}; {source_str}'

    def _log_counts(self, retriever_name:str, title:str, search_results:SearchResultCollection):
        
        result_count = len(search_results.results)
        topic_count = sum([len(search_result.topics) for search_result in search_results.results])
        statement_count = sum([
            
            len(topic.statements)
            for search_result in search_results.results
            for topic in search_result.topics
        ])

        logger.debug(f'[{retriever_name}.{type(self).__name__}] {title}: [results: {result_count}, topics: {topic_count}, statements: {statement_count}]')

    
    def process_results(self, search_results:SearchResultCollection, query:QueryBundle, retriever_name:str) -> SearchResultCollection:
        self._log_counts(retriever_name, 'Before', search_results)
        self._log_results(retriever_name, 'Before', search_results)
        search_results = self._process_results(search_results, query)
        self._log_counts(retriever_name, 'After', search_results)
        self._log_results(retriever_name, 'After', search_results)
        return search_results
    
    @abc.abstractmethod
    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        raise NotImplementedError
