# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import statistics
from typing import List

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs


logger = logging.getLogger(__name__)

class EntityContextProvider():
    
    def __init__(self, graph_store:GraphStore, args:ProcessorArgs):
        self.graph_store = graph_store
        self.args = args
                        
    def get_entity_contexts(self, entities:List[ScoredEntity])  -> List[List[ScoredEntity]]:

        baseline_score = entities[0].score
        upper_score_threshold = baseline_score * self.args.ec_max_score_factor

        
        original_entity_ids = [entity.entity.entityId for entity in entities if entity.score > 0]  
        neighbour_entity_ids = set()
        
        start_entity_ids = set(original_entity_ids) 
        exclude_entity_ids = set(start_entity_ids)

        root_contexts = { entity_id:{} for entity_id in start_entity_ids }
        current_contexts = { k:v for k,v in root_contexts.items() }

        
        
        for limit in range (3, 1, -1):
        
            cypher = f"""
            // expand entities
            MATCH (entity:`__Entity__`)
            -[:`__SUBJECT__`|`__OBJECT__`]->(:`__Fact__`)<-[:`__SUBJECT__`|`__OBJECT__`]-
            (other:`__Entity__`)
            WHERE  {self.graph_store.node_id('entity.entityId')} IN $entityIds
            AND NOT {self.graph_store.node_id('other.entityId')} IN $excludeEntityIds
            WITH entity, other
            MATCH (other)-[r:`__SUBJECT__`|`__OBJECT__`]->()
            WITH entity, other, count(r) AS score ORDER BY score DESC
            RETURN {{
                {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                others: collect(DISTINCT {self.graph_store.node_id('other.entityId')})[0..$limit]
            }} AS result    
            """

            params = {
                'entityIds': list(start_entity_ids),
                'excludeEntityIds': list(exclude_entity_ids),
                'limit': limit
            }
        
            results = self.graph_store.execute_query(cypher, params)

            new_current_contexts = {}

            for result in results:
                start_entity_id = result['result']['entity']['entityId']
                other_entity_ids = result['result']['others']

                for other_entity_id in other_entity_ids:
                    child_context = { }
                    current_contexts[start_entity_id][other_entity_id] = child_context
                    new_current_contexts[other_entity_id] = child_context

            
            other_entity_ids = set([
                other_id
                for result in results
                for other_id in result['result']['others'] 
            ])
            
            neighbour_entity_ids.update(other_entity_ids)

            exclude_entity_ids.update(other_entity_ids)
            start_entity_ids = other_entity_ids

            current_contexts = new_current_contexts

        
        cypher = f"""
        // expand entities: score entities by number of facts
        MATCH (entity:`__Entity__`)-[r:`__SUBJECT__`]->(f:`__Fact__`)
        WHERE {self.graph_store.node_id('entity.entityId')} IN $entityIds
        WITH entity, count(r) AS score
        RETURN {{
            {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
            score: score
        }} AS result
        """

        params = {
            'entityIds': list(neighbour_entity_ids)
        }

        results = self.graph_store.execute_query(cypher, params)

        
        
        neighbour_entities = [
            ScoredEntity.model_validate(result['result'])
            for result in results 
            if result['result']['entity']['entityId'] not in original_entity_ids and result['result']['score'] <= upper_score_threshold and result['result']['score'] >= (baseline_score * self.args.ec_min_score_factor)
        ]
        
        neighbour_entities.sort(key=lambda e:e.score, reverse=True)

        num_addition_entities = self.args.max_keywords - len(entities)

        entities.extend(neighbour_entities[:num_addition_entities])        
 

        all_entities = {
            entity.entity.entityId:entity for entity in entities
        }

        unsorted_contexts = []

        def walk_tree(current_context, d):
            for k,v in d.items():
                context = [c for c in current_context]
                if k in all_entities:
                    context.append(all_entities[k])
                if not v and k in all_entities:
                    unsorted_contexts.append(context)
                else:
                    walk_tree(context, v)

        walk_tree([], root_contexts)

        #contexts.sort(key=lambda context:statistics.mean([c.score for c in context]), reverse=True)

        sorted_contexts = []

        for entity in entities:
            for context in unsorted_contexts:
                if context[0].entity.entityId == entity.entity.entityId:
                    sorted_contexts.append(context)

        

        if logger.isEnabledFor(logging.DEBUG):

            output = [
                str([e.entity.value for e in context])
                for context in sorted_contexts
            ]
            logger.debug('Contexts:\n' + '\n'.join(output))
    
       
        return sorted_contexts

        