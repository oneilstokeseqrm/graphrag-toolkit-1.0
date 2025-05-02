# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
        
import json
import yaml
import logging
import time
from json2xml import json2xml
from typing import Optional, List, Type, Union

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.tenant_id import TenantIdType, to_tenant_id
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.retrieval.prompts import ANSWER_QUESTION_SYSTEM_PROMPT, ANSWER_QUESTION_USER_PROMPT
from graphrag_toolkit.lexical_graph.retrieval.post_processors.bedrock_context_format import BedrockContextFormat
from graphrag_toolkit.lexical_graph.retrieval.retrievers import CompositeTraversalBasedRetriever, SemanticGuidedRetriever
from graphrag_toolkit.lexical_graph.retrieval.retrievers import StatementCosineSimilaritySearch, KeywordRankingSearch, SemanticBeamGraphSearch
from graphrag_toolkit.lexical_graph.retrieval.retrievers import WeightedTraversalBasedRetrieverType, SemanticGuidedRetrieverType
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory, GraphStoreType
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory, VectorStoreType
from graphrag_toolkit.lexical_graph.storage.graph import MultiTenantGraphStore
from graphrag_toolkit.lexical_graph.storage.vector import MultiTenantVectorStore, ReadOnlyVectorStore
from graphrag_toolkit.lexical_graph.storage.vector import to_embedded_query

from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.base.response.schema import Response
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType

logger = logging.getLogger(__name__)

RetrieverType = Union[BaseRetriever, Type[BaseRetriever]]
PostProcessorsType = Union[BaseNodePostprocessor, List[BaseNodePostprocessor]]

class LexicalGraphQueryEngine(BaseQueryEngine):

    @staticmethod
    def for_traversal_based_search(graph_store:GraphStoreType, 
                                   vector_store:VectorStoreType, 
                                   tenant_id:Optional[TenantIdType]=None,
                                   retrievers:Optional[List[WeightedTraversalBasedRetrieverType]]=None,
                                   post_processors:Optional[PostProcessorsType]=None, 
                                   filter_config:FilterConfig=None,
                                   **kwargs):
        
        tenant_id = to_tenant_id(tenant_id)
        
        graph_store =  MultiTenantGraphStore.wrap(GraphStoreFactory.for_graph_store(graph_store), tenant_id) 
        vector_store = ReadOnlyVectorStore.wrap( 
            MultiTenantVectorStore.wrap(VectorStoreFactory.for_vector_store(vector_store), tenant_id)
        )
        
        retriever = CompositeTraversalBasedRetriever(
            graph_store, 
            vector_store, 
            retrievers=retrievers,
            filter_config=filter_config,
            **kwargs
        )
        
        return LexicalGraphQueryEngine(
            graph_store, 
            vector_store,
            tenant_id=tenant_id,
            retriever=retriever,
            post_processors=post_processors,
            context_format='text',
            filter_config=filter_config,
            **kwargs
        )
    
    @staticmethod
    def for_semantic_guided_search(graph_store:GraphStoreType, 
                                   vector_store:VectorStoreType, 
                                   tenant_id:Optional[TenantIdType]=None,
                                   retrievers:Optional[List[SemanticGuidedRetrieverType]]=None,
                                   post_processors:Optional[PostProcessorsType]=None, 
                                   filter_config:FilterConfig=None,
                                   **kwargs):
        
        tenant_id = to_tenant_id(tenant_id)
        
        graph_store =  MultiTenantGraphStore.wrap(GraphStoreFactory.for_graph_store(graph_store), tenant_id) 
        vector_store = ReadOnlyVectorStore.wrap( 
            MultiTenantVectorStore.wrap(VectorStoreFactory.for_vector_store(vector_store), tenant_id)
        )
        
        retrievers = retrievers or [
            StatementCosineSimilaritySearch(
                vector_store=vector_store,
                graph_store=graph_store,
                top_k=50,
                filter_config=filter_config
            ),
            KeywordRankingSearch(
                vector_store=vector_store,
                graph_store=graph_store,
                max_keywords=10,
                filter_config=filter_config
            ),
            SemanticBeamGraphSearch(
                vector_store=vector_store,
                graph_store=graph_store,
                max_depth=8,
                beam_width=100,
                filter_config=filter_config
            )
        ]
        
        retriever = SemanticGuidedRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            retrievers=retrievers,
            share_results=True,
            filter_config=filter_config,
            **kwargs
        ) 
        
        return LexicalGraphQueryEngine(
            graph_store, 
            vector_store,
            tenant_id=tenant_id,
            retriever=retriever,
            post_processors=post_processors,
            context_format='bedrock_xml',
            filter_config=filter_config,
            **kwargs
        )


    def __init__(self, 
                 graph_store:GraphStoreType,
                 vector_store:VectorStoreType,
                 tenant_id:Optional[TenantIdType]=None,
                 llm:LLMCacheType=None,
                 system_prompt:Optional[str]=ANSWER_QUESTION_SYSTEM_PROMPT,
                 user_prompt:Optional[str]=ANSWER_QUESTION_USER_PROMPT,
                 retriever:Optional[RetrieverType]=None,
                 post_processors:Optional[PostProcessorsType]=None,
                 callback_manager: Optional[CallbackManager]=None, 
                 filter_config:FilterConfig=None,
                 **kwargs):
        
        tenant_id = to_tenant_id(tenant_id)
        
        graph_store =  MultiTenantGraphStore.wrap(GraphStoreFactory.for_graph_store(graph_store), tenant_id) 
        vector_store = ReadOnlyVectorStore.wrap( 
            MultiTenantVectorStore.wrap(VectorStoreFactory.for_vector_store(vector_store), tenant_id)
        )

        self.context_format = kwargs.get('context_format', 'json')
        
        self.llm = llm if llm and isinstance(llm, LLMCache) else LLMCache(
            llm=llm or GraphRAGConfig.response_llm,
            enable_cache=GraphRAGConfig.enable_cache
        )
        self.chat_template = ChatPromptTemplate(message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ])

        if retriever:
            if isinstance(retriever, BaseRetriever):
                self.retriever = retriever
            else:
                self.retriever = retriever(graph_store, vector_store, filter_config=filter_config, **kwargs)
        else:
            self.retriever = CompositeTraversalBasedRetriever(graph_store, vector_store, filter_config=filter_config, **kwargs)

        if post_processors:
            self.post_processors = post_processors if isinstance(post_processors, list) else [post_processors]
        else:
            self.post_processors = []

        if self.context_format == 'bedrock_xml':
            self.post_processors.append(BedrockContextFormat())

        if callback_manager:
            for post_processor in self.post_processors:
                post_processor.callback_manager = callback_manager
        

        super().__init__(callback_manager)

    def _generate_response(
        self, 
        query_bundle: QueryBundle, 
        context: str
    ) -> str:
        try:
            response = self.llm.predict(
                prompt=self.chat_template,
                query=query_bundle.query_str,
                search_results=context
            )
            return response
        except Exception:
            logger.exception(f'Error answering query [query: {query_bundle.query_str}, context: {context}]')
            raise
            
    def _format_as_text(self, json_results):
        lines = []
        for json_result in json_results:
            lines.append(f"""## {json_result['topic']}""")
            lines.append(' '.join([s for s in json_result['statements']]))
            lines.append(f"""[Source: {json_result['source']}]""")
            lines.append('\n')
        return '\n'.join(lines)
    
    def _format_context(self, search_results:List[NodeWithScore], context_format:str='json'):

        if context_format == 'bedrock_xml':
            return '\n'.join([result.text for result in search_results])
        
        json_results = [json.loads(result.text) for result in search_results]
        
        data = None
        
        if context_format == 'yaml':
            data = yaml.dump(json_results, sort_keys=False)
        elif context_format == 'xml':
            data = json2xml.Json2xml(json_results, attr_type=False).to_xml()
        elif context_format == 'text':
            data = self._format_as_text(json_results)
        else:
            data = json.dumps(json_results, indent=2)
        
        return data
    
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:

        query_bundle = QueryBundle(query_bundle) if isinstance(query_bundle, str) else query_bundle

        query_bundle = to_embedded_query(query_bundle, GraphRAGConfig.embed_model)
                
        results = self.retriever.retrieve(query_bundle)

        for post_processor in self.post_processors:
            results = post_processor.postprocess_nodes(results, query_bundle)

        return results

 
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:

        try:
        
            start = time.time()

            query_bundle = to_embedded_query(query_bundle, GraphRAGConfig.embed_model)
                
            results = self.retriever.retrieve(query_bundle)

            end_retrieve = time.time()

            for post_processor in self.post_processors:
                results = post_processor.postprocess_nodes(results, query_bundle)

            end_postprocessing = time.time()

            context = self._format_context(results, self.context_format)
            answer = self._generate_response(query_bundle, context)
            
            end = time.time()

            retrieve_ms = (end_retrieve-start) * 1000
            postprocess_ms = (end_postprocessing - end_retrieve) * 1000
            answer_ms = (end-end_retrieve) * 1000
            total_ms = (end-start) * 1000

            metadata = {
                'retrieve_ms': retrieve_ms,
                'postprocessing_ms': postprocess_ms,
                'answer_ms': answer_ms,
                'total_ms': total_ms,
                'context_format': self.context_format,
                'retriever': f'{type(self.retriever).__name__}: {self.retriever.__dict__}',
                'query': query_bundle.query_str,
                'postprocessors': [type(p).__name__ for p in self.post_processors],
                'context': context,
                'num_source_nodes': len(results)
            }

            return Response(
                response=answer,
                source_nodes=results,
                metadata=metadata
            )
        except Exception as e:
            logger.exception('Error in query processing')
            raise
        
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass
        
    def _get_prompts(self) -> PromptDictType:
        pass

    def _get_prompt_modules(self) -> PromptMixinType:
        pass

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        pass 