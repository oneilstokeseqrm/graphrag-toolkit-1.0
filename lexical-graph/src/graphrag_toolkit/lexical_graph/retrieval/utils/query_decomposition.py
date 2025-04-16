# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List

from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.retrieval.prompts import EXTRACT_SUBQUERIES_PROMPT, IDENTIFY_MULTIPART_QUESTION_PROMPT

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

SINGLE_QUESTION_THRESHOLD = 25

class QueryDecomposition():
    def __init__(self,
                 llm:LLMCacheType=None, 
                 identify_multipart_question_template=IDENTIFY_MULTIPART_QUESTION_PROMPT,
                 extract_subqueries_template=EXTRACT_SUBQUERIES_PROMPT,
                 max_subqueries=2):
        self.llm = llm if llm and isinstance(llm, LLMCache) else LLMCache(
            llm=llm or GraphRAGConfig.response_llm,
            enable_cache=GraphRAGConfig.enable_cache
        )
        self.identify_multipart_question_template = identify_multipart_question_template
        self.extract_subqueries_template = extract_subqueries_template
        self.max_subqueries = max_subqueries

    def _extract_subqueries(self, s:str) -> List[QueryBundle]:

        response = self.llm.predict(
                PromptTemplate(template=self.extract_subqueries_template),
                question=s,
                max_subqueries=self.max_subqueries
            )

        return [QueryBundle(query_str=s) for s in response.split('\n') if s]
    
    def _is_multipart_question(self, s:str):

        response = self.llm.predict(
                PromptTemplate(template=self.identify_multipart_question_template),
                question=s
            )

        return response.lower().startswith('no')


    def decompose_query(self, query_bundle: QueryBundle) -> List[QueryBundle]:

        subqueries = [query_bundle]
        
        original_query = query_bundle.query_str

        if len(original_query.split()) > SINGLE_QUESTION_THRESHOLD:
            if self._is_multipart_question(original_query):
                subqueries = self._extract_subqueries(original_query)

        logger.debug(f'Subqueries: {subqueries}')
                 
        return subqueries