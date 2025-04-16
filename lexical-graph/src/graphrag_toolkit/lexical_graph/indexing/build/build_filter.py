# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_BUILD_FILTER = lambda s: False

class BuildFilter():
    def __init__(self, topic_filter_fn:Callable[[str], bool]=None, statement_filter_fn:Callable[[str], bool]=None):
        self.topic_filter_fn = topic_filter_fn or DEFAULT_BUILD_FILTER
        self.statement_filter_fn = statement_filter_fn or DEFAULT_BUILD_FILTER

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

