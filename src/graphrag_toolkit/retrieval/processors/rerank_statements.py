# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import tfidf_matcher as tm
from typing import List, Dict
from dateutil.parser import parse

from graphrag_toolkit import GraphRAGConfig
from graphrag_toolkit.retrieval.model import Source
from graphrag_toolkit.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.retrieval.post_processors import SentenceReranker
from graphrag_toolkit.retrieval.model import SearchResultCollection, SearchResult, Topic, ScoredEntity

from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.node_parser import TokenTextSplitter

logger = logging.getLogger(__name__)

def default_reranking_source_metadata_fn(source:Source):
    def format_value(s):
        try: 
            date = parse(s, fuzzy=False)
            return date.strftime("%B %-d, %Y")
        except ValueError:
            if s.startswith('http'):
                return ''
            else:
                return s
    return ', '.join([format_value(v) for v in source.metadata.values()])

class RerankStatements(ProcessorBase):
    def __init__(self, args:ProcessorArgs, reranking_model=None):
        self.reranking_model = reranking_model or GraphRAGConfig.reranking_model
        super().__init__(args)
        self.reranking_source_metadata_fn = self.args.reranking_source_metadata_fn or default_reranking_source_metadata_fn

    def _score_values_with_tfidf(self, values:List[str], query:QueryBundle, entities:List[ScoredEntity]):

        logger.debug('Reranking with tfidf')

        splitter = TokenTextSplitter(chunk_size=25, chunk_overlap=5)
        match_values = splitter.split_text(query.query_str)

        extras = set([
            entity.entity.value
            for entity in entities
        ])

        if extras:
            match_values.append(', '.join(extras))

        logger.debug(f'Match values: {match_values}')
 
        values_to_score = values.copy()
        
        limit =  len(values_to_score)
        if self.args.max_statements:
            limit = min(self.args.max_statements, limit)

        while len(values_to_score) <= limit:
            values_to_score.append('')

        scored_values = {}

        try:
            
            matcher_results = tm.matcher(match_values, values_to_score, limit, 3)

            max_i = len(matcher_results.columns)
        
            for row_index in range(0, len(match_values)):
                for col_index in range(1, max_i, 3) :
                    value = matcher_results.iloc[row_index, col_index]
                    score = matcher_results.iloc[row_index, col_index+1]
                    if value not in scored_values:
                        scored_values[value] = score
                    else:
                        scored_values[value] = max(scored_values[value], score)
        except ValueError:
            scored_values = {v: 0.0 for v in values_to_score if v}
        
                
        sorted_scored_values = dict(sorted(scored_values.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_scored_values

    def _score_values(self, values:List[str], query:QueryBundle, entities:List[ScoredEntity]) -> Dict[str, float]:
            
        logger.debug('Reranking with SentenceReranker')

        reranker = SentenceReranker(model=self.reranking_model, top_n=self.args.max_statements or len(values))

        rank_query = (
            query 
            if not entities 
            else QueryBundle(query_str=f'{query.query_str} (keywords: {", ".join(set([entity.entity.value for entity in entities]))})')
        )

        reranked_values = reranker.postprocess_nodes(
            [
                NodeWithScore(node=TextNode(text=value), score=0.0)
                for value in values
            ],
            rank_query
        )

        return {
            reranked_value.text : reranked_value.score
            for reranked_value in reranked_values
        }

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:

        if not self.args.reranker or self.args.reranker.lower() == 'none':
            return search_results
       
        values_to_score = []
        
        for search_result in search_results.results:
            source_str = self.reranking_source_metadata_fn(search_result.source)
            for topic in search_result.topics:
                topic_str = topic.topic
                for statement in topic.statements:
                    statement_str = statement.statement_str
                    values_to_score.append(self._format_statement_context(source_str, topic_str, statement_str))
        
        start = time.time()

        scored_values = None
        if self.args.reranker.lower() == 'model':
            scored_values = self._score_values(values_to_score, query, search_results.entities)
        else:
            scored_values = self._score_values_with_tfidf(values_to_score, query, search_results.entities)

        end = time.time()

        rerank_ms = (end-start) * 1000

        logger.debug(f'Rerank duration: {rerank_ms:.2f}ms')

        processor_name = type(self).__name__
        if processor_name in self.args.debug_results and logger.isEnabledFor(logging.DEBUG):
            logger.debug('Scored values:\n' + '\n--------------\n'.join([str(scored_value) for scored_value in scored_values.items()]))

        def rerank_statements(topic:Topic, source_str:str):
            topic_str = topic.topic
            surviving_statements = []
            for statement in topic.statements:
                statement_str = statement.statement_str
                key = self._format_statement_context(source_str, topic_str, statement_str)
                if key in scored_values:
                    statement.score = round(float(scored_values[key]), 4)
                    surviving_statements.append(statement)
            topic.statements = sorted(surviving_statements, key=lambda x: x.score, reverse=True)
            return topic

        def rerank_search_result(index:int, search_result:SearchResult):
            source_str = self.reranking_source_metadata_fn(search_result.source)
            return self._apply_to_topics(search_result, rerank_statements, source_str=source_str)
        
        return self._apply_to_search_results(search_results, rerank_search_result)


