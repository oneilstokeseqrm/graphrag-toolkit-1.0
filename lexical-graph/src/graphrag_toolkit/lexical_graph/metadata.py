# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Union
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

MetadataFiltersType = Union[MetadataFilters, MetadataFilter, List[MetadataFilter]]

class FilterConfig():
    def __init__(self, source_filters:Optional[MetadataFiltersType]=None):
        if not source_filters:
            self.source_filters = None
        elif isinstance(source_filters, MetadataFilters):
            self.source_filters = source_filters
        elif isinstance(source_filters, MetadataFilter):
            self.source_filters = MetadataFilters(filters=[source_filters])
        elif isinstance(source_filters, list):
            self.source_filters = MetadataFilters(filters=source_filters)
        else:
            raise ValueError(f'Invalid source filters type: {type(source_filters)}')