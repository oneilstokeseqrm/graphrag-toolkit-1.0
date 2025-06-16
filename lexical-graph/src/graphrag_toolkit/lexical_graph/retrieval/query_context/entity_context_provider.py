# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import List, Dict

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs
from graphrag_toolkit.lexical_graph.utils.tfidf_utils import score_values

from llama_index.core.schema import QueryBundle


logger = logging.getLogger(__name__)

class EntityContextProvider():
    
    def __init__(self, graph_store:GraphStore, args:ProcessorArgs):
        self.graph_store = graph_store
        self.args = args
        
    def _get_entity_id_context_tree(self, entities:List[ScoredEntity]) -> Dict[str, Dict]:
        
        entity_ids = [entity.entity.entityId for entity in entities if entity.score > 0] 
        neighbour_entity_ids = set()
        
        entity_contexts_id_map = { entity_id:{} for entity_id in entity_ids }
        
        for entity_id, entity_id_context in entity_contexts_id_map.items():
            
            start_entity_ids = set([entity_id])
            exclude_entity_ids = set([entity_id])
            current_entity_id_contexts = { entity_id: entity_id_context  }

            for num_neighbours in range (3, 1, -1):

                cypher = f"""
                // expand entities
                MATCH (entity:`__Entity__`)-[:`__RELATION__`]->(other)
                WHERE  {self.graph_store.node_id('entity.entityId')} IN $entityIds
                AND NOT {self.graph_store.node_id('other.entityId')} IN $excludeEntityIds
                WITH entity, other
                MATCH (other)-[r:`__RELATION__`]-()
                WITH entity, other, count(r) AS score ORDER BY score DESC
                RETURN {{
                    {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                    others: collect(DISTINCT {self.graph_store.node_id('other.entityId')})[0..$numNeighbours]
                }} AS result    
                """

                params = {
                    'entityIds': list(start_entity_ids),
                    'excludeEntityIds': list(exclude_entity_ids),
                    'numNeighbours': num_neighbours
                }

                results = self.graph_store.execute_query(cypher, params)

                new_entity_id_contexts = {}

                for result in results:
                    
                    start_entity_id = result['result']['entity']['entityId']
                    other_entity_ids = result['result']['others']

                    for other_entity_id in other_entity_ids:
                        child_context = { }
                        current_entity_id_contexts[start_entity_id][other_entity_id] = child_context
                        new_entity_id_contexts[other_entity_id] = child_context


                other_entity_ids = set([
                    other_id
                    for result in results
                    for other_id in result['result']['others'] 
                ])

                neighbour_entity_ids.update(other_entity_ids)
                start_entity_ids = other_entity_ids

                current_entity_id_contexts = new_entity_id_contexts
                
        return entity_contexts_id_map
    
    def _get_neighbour_entities(self, entity_id_context_tree:Dict[str, Dict], baseline_score:float) -> List[ScoredEntity]:

        neighbour_entity_ids = set()

        def walk_tree(d):
            for _, v in d.items():
                for k, v in v.items():
                    neighbour_entity_ids.add(k)
                    walk_tree(v)

        walk_tree(entity_id_context_tree)
        
        logger.debug(f'neighbour_entity_ids: {list(neighbour_entity_ids)}')

        cypher = f"""
        // expand entities: score entities by number of relations
        MATCH (entity:`__Entity__`)-[r:`__RELATION__`]-()
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

        upper_score_threshold = baseline_score * self.args.ec_max_score_factor
        lower_score_threshhold = baseline_score * self.args.ec_min_score_factor

        
        neighbour_entities = [
            ScoredEntity.model_validate(result['result'])
            for result in results 
            if result['result']['score'] <= upper_score_threshold and result['result']['score'] >= lower_score_threshhold
        ]
        
        neighbour_entities.sort(key=lambda e:e.score, reverse=True)
        
        logger.debug(f'neighbour_entities: {neighbour_entities}')

        return neighbour_entities

        
    def _get_entity_contexts(self, entities:List[ScoredEntity], entity_id_context_tree:Dict[str, Dict], query_bundle:QueryBundle) -> List[List[ScoredEntity]]:
       
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

        walk_tree([], entity_id_context_tree)
        
        
        logger.debug(f'unsorted_contexts: {unsorted_contexts}')

        context_map = {
            ' '.join([e.entity.value.lower() for e in c]): c
            for c in unsorted_contexts
        }
        
        scored_context_map = score_values(list(context_map.keys()), [query_bundle.query_str])

        logger.debug(f'scored_contexts: {scored_context_map}')

        return [
            context_map[k]
            for k, v in scored_context_map.items()
        ]

        

                        
    def get_entity_contexts(self, entities:List[ScoredEntity], query_bundle:QueryBundle)  -> List[List[ScoredEntity]]:

        entity_id_context_tree = self._get_entity_id_context_tree(entities)
        
        neighbour_entities = self._get_neighbour_entities(
            entity_id_context_tree=entity_id_context_tree,
            baseline_score=entities[0].score
        )

        entities.extend(neighbour_entities)     
      
        entity_contexts = self._get_entity_contexts(
            entities=entities,
            entity_id_context_tree=entity_id_context_tree,
            query_bundle=query_bundle
        )

        if logger.isEnabledFor(logging.DEBUG):

            output = [
                str([e.entity.value for e in context])
                for context in entity_contexts
            ]
            logger.debug('Contexts:\n' + '\n'.join(output))
    
       
        return entity_contexts