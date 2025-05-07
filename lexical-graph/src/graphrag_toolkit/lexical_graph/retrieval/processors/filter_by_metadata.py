# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Any, Dict
from dateutil.parser import parse

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult

from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator

def apply_filter_operator(
        operator: FilterOperator, metadata_value: Any, value: Any 
    ) -> bool:
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
        
def type_name_for_value(value:Any) -> str:
    
    if isinstance(value, list):
        raise ValueError(f'Unsupported value type: {type(value)}')
    
    if isinstance(value, int):
        return 'int'
    elif isinstance(value, float):
        return 'float'
    else:
        try:
            parse(value, fuzzy=False)
            return 'timestamp'
        except ValueError as e:
            return 'text'

def formatter_for_type(type_name:str) -> Callable[[Any], str]:
    if type_name == 'text':
        return lambda x: x
    elif type_name == 'timestamp':
        return lambda x: parse(x, fuzzy=False).isoformat()
    elif type_name == 'int':
        return lambda x:int(x)
    elif type_name == 'float':
        return lambda x:float(x)
    else:
        raise ValueError(f'Unsupported type name: {type_name}')
        
def apply_filters(filter_config:FilterConfig, metadata:Dict[str, Any]) -> bool:
    
    results = []
    
    if filter_config is None or filter_config.source_filters is None:
        return True
    
    for f in filter_config.source_filters.filters:
        metadata_value = metadata.get(f.key, None)
        if f.operator == FilterOperator.IS_EMPTY:
            result = (
                metadata_value is None
                or metadata_value == ''
                or metadata_value == []
            )
        else:
            type_name = type_name_for_value(f.value)
            formatter = formatter_for_type(type_name)
            value = formatter(f.value)
            metadata_value = formatter(metadata_value)
            result = apply_filter_operator(
                operator=f.operator,
                metadata_value=metadata_value,
                value=value  
            )
            
        results.append(result)
        
    filter_condition = filter_config.source_filters.condition

    if filter_condition == FilterCondition.AND:
        return all(results)
    elif filter_condition == FilterCondition.OR:
        return any(results)
    else:
        raise ValueError(f'Unsupported filters condition: {filter_condition}')

class FilterByMetadata(ProcessorBase):
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        super().__init__(args, filter_config)
        
    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        def filter_search_result(index:int, search_result:SearchResult):           
            return search_result if apply_filters(self.filter_config, search_result.source.metadata) else None

        return self._apply_to_search_results(search_results, filter_search_result)