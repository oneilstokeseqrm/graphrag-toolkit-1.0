# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult

from llama_index.core.schema import QueryBundle

class DedupResults(ProcessorBase):
    def __init__(self, args:ProcessorArgs):
        super().__init__(args)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        deduped_results:Dict[str, SearchResult] = {}

        for search_result in search_results.results:
            source_id = search_result.source.sourceId

            if source_id not in deduped_results:
                deduped_results[source_id] = search_result
                continue
            else:
                deduped_result = deduped_results[source_id]
                for topic in search_result.topics:
                    existing_topic = next((x for x in deduped_result.topics if x.topic == topic.topic), None)
                    if not existing_topic:
                        deduped_result.topics.append(topic)
                        continue
                    else:
                        for chunk in topic.chunks:
                            existing_chunk = next((x for x in existing_topic.chunks if x.chunkId == chunk.chunkId), None)
                            if not existing_chunk:
                                existing_topic.chunks.append(chunk)
                        for statement in topic.statements:
                            existing_statement = next((x for x in existing_topic.statements if x.statement == statement.statement), None)
                            if not existing_statement:
                                existing_topic.statements.append(statement)
                            else:
                                existing_statement.score += statement.score
                        
        for search_result in search_results.results:
            for topic in search_result.topics:
                topic.statements = sorted(topic.statements, key=lambda x: x.score, reverse=True)

        search_results = search_results.with_new_results(results=[r for r in deduped_results.values()])
        
        return search_results


