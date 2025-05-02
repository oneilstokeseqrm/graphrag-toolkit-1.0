# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Union
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

MetadataFiltersType = Union[MetadataFilters, MetadataFilter, List[MetadataFilter]]

class FilterConfig():
    def __init__(self, filters:Optional[MetadataFiltersType]=None):
        if not filters:
            self.filters = None
        elif isinstance(filters, MetadataFilters):
            self.filters = filters
        elif isinstance(filters, MetadataFilter):
            self.filters = MetadataFilters(filters=[filters])
        elif isinstance(filters, list):
            self.filters = self.filters = MetadataFilters(filters=filters)
        else:
            raise ValueError(f'Invalid filters type: {type(filters)}')