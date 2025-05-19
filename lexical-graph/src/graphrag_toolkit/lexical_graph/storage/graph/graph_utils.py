# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import re
import string

from graphrag_toolkit.lexical_graph.storage.graph.graph_store import NodeId

SEARCH_STRING_PATTERN = re.compile(r'([^\s\w]|_)+')

def search_string_from(value:str):
    value = SEARCH_STRING_PATTERN.sub('', value)
    while '  ' in value:
        value = value.replace('  ', ' ')
    return value.lower()

def label_from(value:str):
    value = SEARCH_STRING_PATTERN.sub(' ', value)
    return string.capwords(value).replace(' ', '')

def relationship_name_from(value:str):
    return ''.join([ c if c.isalnum() else '_' for c in value ]).upper()

def node_result(node_ref:str, 
                node_id:Optional[NodeId]=None, 
                properties:Optional[List[str]]=['*'], 
                key_name:Optional[str]=None):
    
    key = key_name or node_ref
    
    property_selectors = []
    
    if node_id:
        if node_id.is_property_based:
            if node_id.key not in properties and '*' not in properties:
                property_selectors.append(f'.{node_id.key}')
        else:
            property_selectors.append(f'{node_id.key}: {node_id}')
    
    property_selectors.extend(['.{}'.format(p) for p in properties])
        
    return f'{key}: {node_ref}{{{", ".join(property_selectors)}}}'
    