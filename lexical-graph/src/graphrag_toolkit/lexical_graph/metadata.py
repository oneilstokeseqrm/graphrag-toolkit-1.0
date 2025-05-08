# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Any, Dict, List, Optional, Union
from dateutil.parser import parse

from graphrag_toolkit.lexical_graph import GraphRAGConfig

from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

logger = logging.getLogger(__name__)

MetadataFiltersType = Union[MetadataFilters, MetadataFilter, List[MetadataFilter]]

def is_datetime_key(key):
    return key.endswith(tuple(GraphRAGConfig.metadata_datetime_suffixes))

def format_datetime(s):
    return parse(s, fuzzy=False).isoformat()

def type_name_for_key_value(key:str, value:Any) -> str:

    if isinstance(value, list):
        raise ValueError(f'Unsupported value type: {type(value)}')
    
    if isinstance(value, int):
        return 'int'
    elif isinstance(value, float):
        return 'float'
    else:
        if is_datetime_key(key):
            try:
                parse(value, fuzzy=False)
                return 'timestamp'
            except ValueError as e:
                return 'text'
        else:
            return 'text'
        
def formatter_for_type(type_name:str) -> Callable[[Any], str]:
        if type_name == 'text':
            return lambda x: x
        elif type_name == 'timestamp':
            return lambda x: format_datetime(x)
        elif type_name == 'int':
            return lambda x:int(x)
        elif type_name == 'float':
            return lambda x:float(x)
        else:
            raise ValueError(f'Unsupported type name: {type_name}')

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
        
        self.source_metadata_dictionary_filter_fn = DictionaryFilter(self.source_filters) if self.source_filters else lambda x:True

    def filter_source_metadata_dictionary(self, d:Dict[str, Any]) -> bool:
        result = self.source_metadata_dictionary_filter_fn(d)
        logger.debug(f'filter result: [{str(d)}: {result}]')
        return result
        
class DictionaryFilter():
    def __init__(self, metadata_filters:MetadataFilters):
        self.metadata_filters = metadata_filters

    def _apply_filter_operator(self, operator: FilterOperator, metadata_value: Any, value: Any) -> bool:
            if metadata_value is None:
                return False
            if operator == FilterOperator.EQ:
                return metadata_value == value
            if operator == FilterOperator.NE:
                return metadata_value != value
            if operator == FilterOperator.GT:
                return metadata_value > value
            if operator == FilterOperator.GTE:
                return metadata_value >= value
            if operator == FilterOperator.LT:
                return metadata_value < value
            if operator == FilterOperator.LTE:
                return metadata_value <= value
            if operator == FilterOperator.IN:
                return metadata_value in value
            if operator == FilterOperator.NIN:
                return metadata_value not in value
            if operator == FilterOperator.CONTAINS:
                return value in metadata_value
            if operator == FilterOperator.TEXT_MATCH:
                return value.lower() in metadata_value.lower()
            if operator == FilterOperator.ALL:
                return all(val in metadata_value for val in value)
            if operator == FilterOperator.ANY:
                return any(val in metadata_value for val in value)
            raise ValueError(f'Unsupported filter operator: {operator}')

    
    
    def _apply_metadata_filters_recursive(self, metadata_filters:MetadataFilters, metadata:Dict[str, Any]) -> bool:
        
        results:List[bool] = []

        def get_filter_result(f:MetadataFilter, metadata:Dict[str, Any]):
            metadata_value = metadata.get(f.key, None)
            if f.operator == FilterOperator.IS_EMPTY:
                return (
                    metadata_value is None
                    or metadata_value == ''
                    or metadata_value == []
                )
            else:
                type_name = type_name_for_key_value(f.key, f.value)
                formatter = formatter_for_type(type_name)
                value = formatter(f.value)
                metadata_value = formatter(metadata_value)
                return self._apply_filter_operator(
                    operator=f.operator,
                    metadata_value=metadata_value,
                    value=value  
                )

        for metadata_filter in metadata_filters.filters:
            if isinstance(metadata_filter, MetadataFilter):
                if metadata_filters.condition == FilterCondition.NOT:
                    raise ValueError(f'Expected MetadataFilters for FilterCondition.NOT, but found MetadataFilter')
                results.append(get_filter_result(metadata_filter, metadata))
            elif isinstance(metadata_filter, MetadataFilters):
                results.append(self._apply_metadata_filters_recursive(metadata_filter, metadata))
            else:
                raise ValueError(f'Invalid metadata filter type: {type(metadata_filter)}')
            
        if metadata_filters.condition == FilterCondition.NOT:
            return not all(results)
        elif metadata_filters.condition == FilterCondition.AND:
            return all(results)
        elif metadata_filters.condition == FilterCondition.OR:
            return any(results)
        else:
            raise ValueError(f'Unsupported filters condition: {metadata_filters.condition}')

    def __call__(self, metadata:Dict[str, Any]) -> bool:
        return self._apply_metadata_filters_recursive(self.metadata_filters, metadata)