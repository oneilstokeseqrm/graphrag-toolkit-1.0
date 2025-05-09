# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Dict, Any, Optional

from graphrag_toolkit.lexical_graph.metadata import MetadataFiltersType, FilterConfig
from llama_index.core.bridge.pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_BUILD_FILTER = lambda s: False

class BuildFilters(BaseModel):

    topic_filter_fn:Callable[[str], bool]
    statement_filter_fn:Callable[[str], bool]
    source_filters:FilterConfig

    def __init__(self, 
                 topic_filter_fn:Callable[[str], bool]=None, 
                 statement_filter_fn:Callable[[str], bool]=None,
                 source_filters:Optional[MetadataFiltersType]=None
        ):
        super().__init__(
            topic_filter_fn = topic_filter_fn or DEFAULT_BUILD_FILTER,
            statement_filter_fn = statement_filter_fn or DEFAULT_BUILD_FILTER,
            source_filters = FilterConfig(source_filters)
        )

    def ignore_topic(self, topic:str) -> bool:
        result = self.topic_filter_fn(topic)
        if result:
            logger.debug(f'Ignore topic: {topic}')
        return result
    
    def ignore_statement(self, statement:str) -> bool:
        result = self.statement_filter_fn(statement)
        if result:
            logger.debug(f'Ignore statement: {statement}')
        return result
    
    def filter_source_metadata_dictionary(self, d:Dict[str, Any]) -> bool:
        return self.source_filters.filter_source_metadata_dictionary(d)

