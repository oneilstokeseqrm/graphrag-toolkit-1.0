# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Dict, Any, Annotated, Optional
from pydantic import Field

from graphrag_toolkit.lexical_graph import LexicalGraphQueryEngine
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.summary import GraphSummary, get_domain


def query_tenant_graph(graph_store, vector_store, tenant_id, domain):
    
    description = f'A natural language query related to the domain of {domain}' if domain else 'A natural language query'
    
    def query_graph(query: Annotated[str, Field(description=description)]) -> List[Dict[str, Any]]:
        
        query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
           graph_store, 
            vector_store,
            tenant_id=tenant_id
        )
        
        response = query_engine.retrieve(query)
        
        return [n.metadata for n in response]
        
    return query_graph

def get_tenant_ids(graph_store:GraphStore):
    
    cypher = '''MATCH (n)
    WITH DISTINCT labels(n) as lbls
    WITH split(lbls[0], '__') AS lbl_parts WHERE size(lbl_parts) > 4
    RETURN DISTINCT lbl_parts[size(lbl_parts) - 2] AS tenant_id
    '''
    results = graph_store.execute_query(cypher)

    return [result['tenant_id'] for result in results]

def create_mcp_server(graph_store:GraphStore, vector_store:VectorStore, tenant_ids:Optional[List[str]]=None):

    try:
        from fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "fastmcp package not found, install with 'pip install fastmcp'"
        ) from e

    mcp = FastMCP(name='LexicalGraphServer')

    graph_summary = GraphSummary(graph_store)

    if not tenant_ids:
        tenant_ids = [None]
        tenant_ids.extend(get_tenant_ids(graph_store))
    
    for tenant_id in tenant_ids:
        
        summary = graph_summary.create_graph_summary(tenant_id)

        if summary:
            domain = get_domain(summary)
            
            mcp.add_tool(
                fn=query_tenant_graph(graph_store, vector_store, tenant_id, domain),
                name = tenant_id if tenant_id else 'default_',
                description = summary
            )

    return mcp


