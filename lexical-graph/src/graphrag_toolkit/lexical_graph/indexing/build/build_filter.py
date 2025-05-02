# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Dict, Any, Optional
from dateutil.parser import parse

logger = logging.getLogger(__name__)

DEFAULT_BUILD_FILTER = lambda s: False

def default_source_metadata_formatter(metadata:Dict[str, Any]) -> Dict[str, Any]:
    formatted_metadata = {}
    for k, v in metadata.items():
        if k.endswith('_date') or k.endswith('_datetime'):
            try:
                dt = parse(v, fuzzy=False).isoformat()
                formatted_metadata[k] = dt
            except ValueError as e:
                formatted_metadata[k] = v
        else:
            formatted_metadata[k] = v
    return formatted_metadata


class BuildFilter():
    def __init__(self, 
                 topic_filter_fn:Callable[[str], bool]=None, 
                 statement_filter_fn:Callable[[str], bool]=None,
                 source_metadata_formatter_fn:Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]=None,
        ):
        self.topic_filter_fn = topic_filter_fn or DEFAULT_BUILD_FILTER
        self.statement_filter_fn = statement_filter_fn or DEFAULT_BUILD_FILTER
        self.source_metadata_formatter_fn = source_metadata_formatter_fn or default_source_metadata_formatter

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

