# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field, AliasChoices
from typing import List, Optional, Union, Dict

class Statement(BaseModel):
    model_config = ConfigDict(strict=True)

    statementId:Optional[str]=None
    statement:str
    facts:List[str]=[]
    details:Optional[str]=None
    chunkId:Optional[str]=None
    score:Optional[float]=None
    statement_str:Optional[str]=None

StatementType = Union[Statement, str]

class Chunk(BaseModel):
    model_config = ConfigDict(strict=True)

    chunkId:str
    value:Optional[str]=None
    score:Optional[float]=None

class Topic(BaseModel):
    model_config = ConfigDict(strict=True)

    topic:str
    chunks:List[Chunk]=[]
    statements:List[StatementType]=[]  

class Source(BaseModel):
    model_config = ConfigDict(strict=True)
    
    sourceId:str
    metadata:Dict[str, str]={}

SourceType = Union[str, Source]

class SearchResult(BaseModel):
    model_config = ConfigDict(strict=True)

    source:SourceType
    topics:List[Topic]=[]
    topic:Optional[str]=None
    statements:List[StatementType]=[]
    score:Optional[float]=None

class Entity(BaseModel):
    model_config = ConfigDict(strict=True)

    entityId:str
    value:str
    classification:str = Field(alias=AliasChoices('class', 'classification'))

class ScoredEntity(BaseModel):
    model_config = ConfigDict(strict=True)

    entity:Entity
    score:float

class SearchResultCollection(BaseModel):

    model_config = ConfigDict(strict=True)

    results: List[SearchResult]=[]
    entities: List[ScoredEntity]=[]

    def add_search_result(self, result:SearchResult):
        self.results.append(result)

    def add_entity(self, entity:ScoredEntity):
        if self.entities is None:
            self.entities = []
        existing_entity = next((x for x in self.entities if x.entity.entityId == entity.entity), None)
        if existing_entity:
            existing_entity.score += entity.score
        else:
            self.entities.append(entity)

    def with_new_results(self, results:List[SearchResult]):
        self.results = results
        return self



