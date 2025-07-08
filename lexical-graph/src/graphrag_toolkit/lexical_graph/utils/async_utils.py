# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import multiprocessing
import warnings
import llama_index.core.async_utils
from typing import Any, Coroutine
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig

logger = logging.getLogger(__name__)

def set_asyncio_run(msg):
    def _asyncio_run(coro: Coroutine) -> Any:
        logger.log(logging.WARNING, msg)       
        try:
            loop = asyncio.get_event_loop()

            return loop.run_until_complete(coro)

        except RuntimeError as e:
            try:
                return asyncio.run(coro)
            except RuntimeError as e:
                raise RuntimeError(
                    "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                    "Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc."
                )
    return _asyncio_run

try:
    loop = asyncio.get_event_loop()
    if loop.is_running:
        num_workers = GraphRAGConfig.extraction_num_workers
        num_threads_per_worker = GraphRAGConfig.extraction_num_threads_per_worker
        num_cpus =  multiprocessing.cpu_count()
        if num_workers * num_threads_per_worker > num_cpus:
            warning = (f'Setting `asyncio_run()` to run coroutines on existing event loop because number of workers ({num_workers}) * number of threads per worker ({num_threads_per_worker}) > number of CPUs ({num_cpus}). To remediate, adjust `GraphRAGConfig.extraction_num_workers` and/or `GraphRAGConfig.extraction_num_threads_per_worker`.')
            llama_index.core.async_utils.asyncio_run = set_asyncio_run(warning)
except RuntimeError as e:
    pass  
