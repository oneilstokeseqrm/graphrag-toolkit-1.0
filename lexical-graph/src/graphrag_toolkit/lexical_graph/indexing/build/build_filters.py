# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Dict, Any, Optional
from dateutil.parser import parse

from graphrag_toolkit.lexical_graph.metadata import is_datetime_key, format_datetime, type_name_for_key_value, formatter_for_type, MetadataFiltersType, FilterConfig

logger = logging.getLogger(__name__)

DEFAULT_BUILD_FILTER = lambda s: False

def default_source_metadata_formatter(metadata:Dict[str, Any]) -> Dict[str, Any]:
    formatted_metadata = {}
    for k, v in metadata.items():
        try:
            type_name = type_name_for_key_value(k, v)
            formatter = formatter_for_type(type_name)
            value = formatter(v)
            formatted_metadata[k] = value
        except ValueError as e:
            formatted_metadata[k] = v
    return formatted_metadata


class BuildFilters():
    def __init__(self, 
                 topic_filter_fn:Callable[[str], bool]=None, 
                 statement_filter_fn:Callable[[str], bool]=None,
                 source_metadata_formatter_fn:Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]=None,
                 source_filters:Optional[MetadataFiltersType]=None
        ):
        self.topic_filter_fn = topic_filter_fn or DEFAULT_BUILD_FILTER
        self.statement_filter_fn = statement_filter_fn or DEFAULT_BUILD_FILTER
        self.source_metadata_formatter_fn = source_metadata_formatter_fn or default_source_metadata_formatter
        self.filter_config = FilterConfig(source_filters)

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
    
    def format_metadata(self, metadata:Dict[str, Any]) -> Dict[str, Any]:
        return self.source_metadata_formatter_fn(metadata=metadata)
    
    def filter_source_metadata_dictionary(self, d:Dict[str, Any]) -> bool:
        return self.filter_config.filter_source_metadata_dictionary(d)

