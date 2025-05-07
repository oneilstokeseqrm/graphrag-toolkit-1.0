# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Callable, Union, List, Any
from string import Template

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Source

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

SourceInfoTemplateType = Union[str, Template]

def default_source_formatter_fn(source:Source):
    if source.metadata:
        source_strs = [str(v) for v in source.metadata.values()]
        source_strs.sort(key=len, reverse=True)
        return ' '.join(source_strs)
    else:
        return source.sourceId

def source_info_template(template:SourceInfoTemplateType) -> Callable[[Dict[str, Any]], str]:
    t = template if isinstance(template, Template) else Template(template)
    def source_info_template_fn(source:Source) -> str:
        return t.safe_substitute(source.metadata)
    return source_info_template_fn

def source_info_keys(keys:List[str]) -> Callable[[Dict[str, Any]], str]:
    def source_info_keys_fn(source:Source) -> str:
        for key in keys:
            if key in source.metadata:
                return source.metadata[key]
        return None
    return source_info_keys_fn

class FormatSources(ProcessorBase):
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        super().__init__(args, filter_config)

        formatter = self.args.source_formatter or default_source_formatter_fn

        fn = None

        if isinstance(formatter, str):
            fn = source_info_template(formatter) if '$' in formatter else source_info_keys([formatter])
        elif isinstance(formatter, list):
            fn = source_info_keys(formatter)
        elif isinstance(formatter, Template):
            fn = source_info_template(formatter)
        elif isinstance(formatter, Callable):
            fn = formatter
        else:
            fn = default_source_formatter_fn

        self.formatter_fn = fn

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        
        def format_source(index:int, search_result:SearchResult):
            try:
                search_result.source = self.formatter_fn(search_result.source)
            except Exception as e:
                logger.error(f'Error while formatting source: {str(e)}')
            return search_result
        
        return self._apply_to_search_results(search_results, format_source)


