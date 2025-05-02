# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import string
import logging
import json
from typing import Any, List, Optional, Callable
from dateutil.parser import parse

from graphrag_toolkit.lexical_graph import IndexError
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore, MultiTenantGraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.storage.graph.neptune_graph_stores import NeptuneAnalyticsClient
from graphrag_toolkit.lexical_graph.storage.vector import VectorIndex, VectorIndexFactoryMethod, to_embedded_query

from llama_index.core.indices.utils import embed_nodes
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilters, MetadataFilter

logger = logging.getLogger(__name__)

NEPTUNE_ANALYTICS = 'neptune-graph://'

def to_opencypher_operator(operator: FilterOperator) -> tuple[str, Callable[[Any], str]]:
    
    default_value_formatter = lambda x: x
    
    operator_map = {
        FilterOperator.EQ: ('=', default_value_formatter), 
        FilterOperator.GT: ('>', default_value_formatter), 
        FilterOperator.LT: ('<', default_value_formatter), 
        FilterOperator.NE: ('<>', default_value_formatter), 
        FilterOperator.GTE: ('>=', default_value_formatter), 
        FilterOperator.LTE: ('<=', default_value_formatter), 
        #FilterOperator.IN: ('in', default_value_formatter),  # In array (string or number)
        #FilterOperator.NIN: ('nin', default_value_formatter),  # Not in array (string or number)
        #FilterOperator.ANY: ('any', default_value_formatter),  # Contains any (array of strings)
        #FilterOperator.ALL: ('all', default_value_formatter),  # Contains all (array of strings)
        #FilterOperator.TEXT_MATCH: ('LIKE', lambda x: f"%%{x}%%"),
        #FilterOperator.TEXT_MATCH_INSENSITIVE: ('~*', default_value_formatter),
        #FilterOperator.CONTAINS: ('contains', default_value_formatter),  # metadata array contains value (string or number)
        #FilterOperator.IS_EMPTY: ('is_empty', default_value_formatter),  # the field is not exist or empty (null or empty array)
    }

    if operator not in operator_map:
        raise ValueError(f'Unsupported filter operator: {operator}')
    
    return operator_map[operator]

def type_name_for_value(value:Any) -> str:
    
    if isinstance(value, list):
        raise ValueError(f'Unsupported value type: {type(value)}')
    
    if isinstance(value, int):
        return 'number'
    elif isinstance(value, float):
        return 'number'
    else:
        try:
            parse(value, fuzzy=False)
            return 'timestamp'
        except ValueError as e:
            return 'text'

def formatter_for_type(type_name:str) -> Callable[[Any], str]:
    if type_name == 'text':
        return lambda x: f"'{x}'"
    elif type_name == 'timestamp':
        return lambda x: f"datetime('{parse(x, fuzzy=False).isoformat()}')"
    elif type_name == 'number':
        return lambda x:x
    else:
        raise ValueError(f'Unsupported type name: {type_name}')
        
    
def filters_to_opencypher_where_clause(filters: MetadataFilters) -> str:

    if filters is None:
        return ''

    def to_key(key: str) -> str:
        return f"source.{key}" 
    
    def to_opencypher_where_clause(f: MetadataFilter) -> str:
        key = to_key(f.key)
        type_name = type_name_for_value(f.value)
        type_formatter = formatter_for_type(type_name)
        (operator, operator_formatter) = to_opencypher_operator(f.operator)
        
        return f"({key}) {operator} {type_formatter(operator_formatter(str(f.value)))}"
        

    if len(filters.filters) == 1:
        f = filters.filters[0]
        where_clause = to_opencypher_where_clause(f)
    else:
        if filters.condition == FilterCondition.AND:
            condition = 'AND'
        elif filters.condition == FilterCondition.OR:
            condition = 'OR'
        else:
            raise ValueError(f'Unsupported filters condition: {filters.condition}')
        
        where_clause = f' {condition} '.join([
            f"{to_opencypher_where_clause(f)}\n"
            for f in filters.filters
        ])

    return f'WHERE {where_clause}'

class NeptuneAnalyticsVectorIndexFactory(VectorIndexFactoryMethod):
    def try_create(self, index_names:List[str], vector_index_info:str, **kwargs) -> List[VectorIndex]:
        graph_id = None
        if vector_index_info.startswith(NEPTUNE_ANALYTICS):
            graph_id = vector_index_info[len(NEPTUNE_ANALYTICS):]
            logger.debug(f'Opening Neptune Analytics vector indexes [index_names: {index_names}, graph_id: {graph_id}]')
            return [NeptuneIndex.for_index(index_name, vector_index_info, **kwargs) for index_name in index_names]
        else:
            return None

class NeptuneIndex(VectorIndex):
    
    @staticmethod
    def for_index(index_name, graph_id, embed_model=None, dimensions=None):

        index_name = index_name.lower()
        neptune_client:GraphStore = GraphStoreFactory.for_graph_store(graph_id)
        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions
        id_name = f'{index_name}Id'
        label = f'__{string.capwords(index_name)}__' 
        path = f'({index_name})'
        return_fields = node_result(index_name, neptune_client.node_id(f'{index_name}.{id_name}'))

        if index_name == 'chunk':
            path = '(chunk)-[:`__EXTRACTED_FROM__`]->(source:`__Source__`)'
            return_fields = f"source:{{sourceId: {neptune_client.node_id('source.sourceId')}, {node_result('source', key_name='metadata')}}},\n{node_result('chunk', neptune_client.node_id('chunk.chunkId'), [])}"
        elif index_name == 'statement':
            path = '(statement)-[:`__MENTIONED_IN__`]->(:`__Chunk__`)-[:`__EXTRACTED_FROM__`]->(source:`__Source__`)'
        elif index_name == 'topic':
            path = '(topic)-[:`__MENTIONED_IN__`]->(:`__Chunk__`)-[:`__EXTRACTED_FROM__`]->(source:`__Source__`)'
        else:
            raise ValueError(f'Invalid index name: {index_name}')
            
        return NeptuneIndex(
            index_name=index_name,
            neptune_client=neptune_client,
            embed_model=embed_model,
            dimensions=dimensions,
            id_name=id_name,
            label=label,
            path=path,
            return_fields=return_fields
        ) 


    neptune_client: NeptuneAnalyticsClient
    embed_model: Any
    dimensions: int
    id_name: str
    label: str
    path: str
    return_fields: str

    def _neptune_client(self):
        if self.tenant_id.is_default_tenant:
            return self.neptune_client
        else:
            return MultiTenantGraphStore.wrap(self.neptune_client, tenant_id=self.tenant_id)

    
    def add_embeddings(self, nodes):
        
        for node in nodes:
            node.metadata['index'] = self.underlying_index_name()
            
        if not self.tenant_id.is_default_tenant():
            raise IndexError('NeptuneIndex does not support multi-tenant indexes')
                
        id_to_embed_map = embed_nodes(
            nodes, self.embed_model
        )
        
        for node in nodes:
        
            statement = f"MATCH (n:`{self.label}`) WHERE {self.neptune_client.node_id('n.{self.id_name}')} = $nodeId"
            
            embedding = id_to_embed_map[node.node_id]
            
            query = '\n'.join([
                statement,
                f'WITH n CALL neptune.algo.vectors.upsert(n, {embedding}) YIELD success RETURN success'
            ])
            
            properties = {
                'nodeId': node.node_id,
                'embedding': embedding
            }

            self._neptune_client().execute_query(query, properties)
            
            node.metadata.pop('index', None)
        
        return nodes
    
    def top_k(self, query_bundle:QueryBundle, top_k:int=5, filters:Optional[MetadataFilters]=None):

        if not self.tenant_id.is_default_tenant():
            raise IndexError('NeptuneIndex does not support multi-tenant indexes')

        prefix = json.dumps({'index': self.underlying_index_name()})

        query_bundle = QueryBundle(query_str=f'{prefix} {query_bundle.query_str}') 

        query_bundle = to_embedded_query(query_bundle, self.embed_model)

        tenant_specific_label = self.tenant_id.format_label(self.label).replace('`', '')

        cypher = f'''
        CALL neptune.algo.vectors.topKByEmbedding(
            {query_bundle.embedding},
            {{   
                topK: {top_k * 5},
                concurrency: 4
            }}
        )
        YIELD node, score       
        WITH node as {self.index_name}, score WHERE '{tenant_specific_label}' in labels({self.index_name}) 
        WITH {self.index_name}, score ORDER BY score ASC LIMIT {top_k}
        MATCH {self.path}
        {filters_to_opencypher_where_clause(filters)}
        RETURN {{
            score: score,
            {self.return_fields}
        }} AS result ORDER BY result.score ASC LIMIT {top_k}
        '''

        results = self._neptune_client().execute_query(cypher)
        
        return [result['result'] for result in results]

    def get_embeddings(self, ids:List[str]=[]):

        if not self.tenant_id.is_default_tenant():
            raise IndexError('NeptuneIndex does not support multi-tenant indexes')
        
        all_results = []
        
        for i in set(ids):

            cypher = f'''
            MATCH (n:`{self.label}`)  WHERE {self.neptune_client.node_id('n.{self.id_name}')} = $elementId
            CALL neptune.algo.vectors.get(
                n
            )
            YIELD node, embedding       
            WITH node as {self.index_name}, embedding WHERE '{self.label}' in labels({self.index_name}) 
            MATCH {self.path}
            RETURN {{
                embedding: embedding,
                {self.return_fields}
            }} AS result
            '''
            
            params = {
                'elementId': i
            }
            
            results = self._neptune_client().execute_query(cypher, params)
            
            for result in results:
                all_results.append(result['result'])
        
        return all_results
