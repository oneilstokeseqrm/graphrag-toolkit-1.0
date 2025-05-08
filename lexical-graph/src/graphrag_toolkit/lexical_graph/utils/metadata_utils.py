# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from dateutil.parser import parse
from graphrag_toolkit.lexical_graph import GraphRAGConfig

def is_datetime_key(key):
    return key.endswith(tuple(GraphRAGConfig.metadata_datetime_suffixes))

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

def format_datetime(s):
    return parse(s, fuzzy=False).isoformat()