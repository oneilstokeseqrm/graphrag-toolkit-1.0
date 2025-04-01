# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import abc  
import uuid
from dataclasses import dataclass
from tenacity import Retrying, stop_after_attempt, wait_random
from tenacity import RetryCallState
from typing import Callable, List, Dict, Any, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REDACTED = '**REDACTED**'
NUM_CHARS_IN_DEBUG_RESULTS = 256

def get_log_formatting(args):
    log_formatting = args.pop('log_formatting', RedactedGraphQueryLogFormatting())
    if not isinstance(log_formatting, GraphQueryLogFormatting):
        raise ValueError('log_formatting must be of type GraphQueryLogFormatting')
    return log_formatting

@dataclass
class NodeId:

    key:str
    value:str
    is_property_based:bool = True

    def __str__(self):
        return self.value
    
def format_id(id_name:str):
        parts = id_name.split('.')
        if len(parts) == 1:
            return NodeId(parts[0], parts[0])           
        else:
            return NodeId(parts[1], id_name)
        
@dataclass
class GraphQueryLogEntryParameters:
    query_ref:str
    query:str
    parameters:str
    results:Optional[str]=None

    def format_query_with_query_ref(self, q):
        return f'//query_ref: {self.query_ref}\n{q}'

class GraphQueryLogFormatting(BaseModel):
    @abc.abstractmethod
    def format_log_entry(self, query_ref:str, query:str, parameters:Dict[str,Any]={}, results:Optional[List[Any]]=None) -> GraphQueryLogEntryParameters:
        raise NotImplementedError
    
class RedactedGraphQueryLogFormatting(GraphQueryLogFormatting):
    def format_log_entry(self, query_ref:str, query:str, parameters:Dict[str,Any]={}, results:Optional[List[Any]]=None) -> GraphQueryLogEntryParameters:
        return GraphQueryLogEntryParameters(query_ref=query_ref, query=REDACTED, parameters=REDACTED, results=REDACTED)
    
class NonRedactedGraphQueryLogFormatting(GraphQueryLogFormatting):
    def format_log_entry(self, query_ref:str, query:str, parameters:Dict[str,Any]={}, results:Optional[List[Any]]=None) -> GraphQueryLogEntryParameters:
        results_str = str(results)
        if len(results_str) > NUM_CHARS_IN_DEBUG_RESULTS:
            results_str = f'{results_str[:NUM_CHARS_IN_DEBUG_RESULTS]}... <{len(results_str) - NUM_CHARS_IN_DEBUG_RESULTS} more chars>'
        return GraphQueryLogEntryParameters(query_ref=query_ref, query=query, parameters=str(parameters), results=results_str)

def on_retry_query(
    logger:'logging.Logger',
    log_level:int,
    log_entry_parameters:GraphQueryLogEntryParameters,
    exc_info:bool=False    
) -> Callable[[RetryCallState], None]:
    
    def log_it(retry_state: 'RetryCallState') -> None:
        local_exc_info: BaseException | bool | None

        if retry_state.outcome is None:
            raise RuntimeError('log_it() called before outcome was set')

        if retry_state.next_action is None:
            raise RuntimeError('log_it() called before next_action was set')

        if retry_state.outcome.failed:
            ex = retry_state.outcome.exception()
            verb, value = 'raised', f'{ex.__class__.__name__}: {ex}'

            if exc_info:
                local_exc_info = retry_state.outcome.exception()
            else:
                local_exc_info = False
        else:
            verb, value = 'returned', retry_state.outcome.result()
            local_exc_info = False  # exc_info does not apply when no exception

        logger.log(
            log_level,
            f'[{log_entry_parameters.query_ref}] Retrying query in {retry_state.next_action.sleep} seconds because it {verb} {value} [attempt: {retry_state.attempt_number}, query: {log_entry_parameters.query}, parameters: {log_entry_parameters.parameters}]',
            exc_info=local_exc_info
        )

    return log_it

def on_query_failed(
    logger:'logging.Logger',
    log_level:int,
    max_attempts:int,
    log_entry_parameters:GraphQueryLogEntryParameters,
) -> Callable[['RetryCallState'], None]:
    
    def log_it(retry_state: 'RetryCallState') -> None:
        if retry_state.attempt_number == max_attempts:
            ex: BaseException | bool | None
            if retry_state.outcome.failed:
                ex = retry_state.outcome.exception()
                verb, value = 'raised', f'{ex.__class__.__name__}: {ex}'       
            logger.log(
                log_level,
                f'[{log_entry_parameters.query_ref}] Query failed after {retry_state.attempt_number} retries because it {verb} {value} [query: {log_entry_parameters.query}, parameters: {log_entry_parameters.parameters}]',
                exc_info=ex
            )
        
    return log_it

class GraphStore(BaseModel):

    log_formatting:GraphQueryLogFormatting = Field(default_factory=lambda: RedactedGraphQueryLogFormatting())
    

    def execute_query_with_retry(self, query:str, parameters:Dict[str, Any], max_attempts=3, max_wait=5, **kwargs):
        
        correlation_id = uuid.uuid4().hex[:5]
        if 'correlation_id' in kwargs:
            correlation_id = f'{kwargs["correlation_id"]}/{correlation_id}'
        kwargs['correlation_id'] = correlation_id

        log_entry_parameters = self.log_formatting.format_log_entry(f'{correlation_id}/*', query, parameters)

        attempt_number = 0
        for attempt in Retrying(
            stop=stop_after_attempt(max_attempts), 
            wait=wait_random(min=0, max=max_wait),
            before_sleep=on_retry_query(logger, logging.WARNING, log_entry_parameters), 
            after=on_query_failed(logger, logging.WARNING, max_attempts, log_entry_parameters),
            reraise=True
        ):
            with attempt:
                attempt_number += 1
                attempt.retry_state.attempt_number
                self.execute_query(query, parameters, **kwargs)

    def _logging_prefix(self, query_id:str, correlation_id:Optional[str]=None):
        return f'{correlation_id}/{query_id}' if correlation_id else f'{query_id}' 
    
    def node_id(self, id_name:str) -> NodeId:
        return format_id(id_name)
    
    @abc.abstractmethod
    def execute_query(self, cypher, parameters={}, correlation_id=None) -> Dict[str, Any]:
        raise NotImplementedError

    
class DummyGraphStore(GraphStore):
    def execute_query(self, cypher, parameters={}, correlation_id=None):  
        log_entry_parameters = self.log_formatting.format_log_entry(self._logging_prefix(correlation_id), cypher, parameters)
        logger.debug(f'[{log_entry_parameters.query_ref}] query: {log_entry_parameters.query}, parameters: {log_entry_parameters.parameters}')
        return []
    