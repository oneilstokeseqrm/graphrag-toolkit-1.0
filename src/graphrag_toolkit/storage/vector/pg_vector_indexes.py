# Copyright FalkorDB.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import psycopg2
import numpy as np
import boto3
from pgvector.psycopg2 import register_vector
from typing import List, Sequence, Dict, Any, Optional
from urllib.parse import urlparse

from graphrag_toolkit.config import GraphRAGConfig, EmbeddingType
from graphrag_toolkit.storage.vector import VectorIndex, VectorIndexFactoryMethod, to_embedded_query
from graphrag_toolkit.storage.constants import INDEX_KEY

from llama_index.core.schema import BaseNode, QueryBundle
from llama_index.core.indices.utils import embed_nodes

logger = logging.getLogger(__name__)

POSTGRES = 'postgres://'
POSTGRESQL = 'postgresql://'

class PGVectorIndexFactory(VectorIndexFactoryMethod):
    def try_create(self, index_names:List[str], vector_index_info:str, **kwargs) -> List[VectorIndex]:
        connection_string = None
        if vector_index_info.startswith(POSTGRES) or vector_index_info.startswith(POSTGRESQL):
            connection_string = vector_index_info
        if connection_string:
            logger.debug(f"Opening PostgreSQL vector indexes [index_names: {index_names}, connection_string: {connection_string}]")
            return [PGIndex.for_index(index_name, connection_string, **kwargs) for index_name in index_names]
        else:
            return None

class PGIndex(VectorIndex):

    @staticmethod
    def for_index(index_name:str,
                  connection_string:str,
                  database='postgres',
                  schema_name='graphrag',
                  host:str='localhost',
                  port:int=5432,
                  username:str=None,
                  password:str=None,
                  embed_model:EmbeddingType=None,
                  dimensions:int=None,
                  enable_iam_db_auth=False):
        
        def compute_enable_iam_db_auth(s, default):
            if 'enable_iam_db_auth' in s.lower():
                return 'enable_iam_db_auth=true' in s.lower()
            else:
                return default
        
        parsed = urlparse(connection_string)

        database = parsed.path[1:] if parsed.path else database
        host = parsed.hostname or host
        port = parsed.port or port
        username = parsed.username or username
        password = parsed.password or password
        enable_iam_db_auth = compute_enable_iam_db_auth(parsed.query, enable_iam_db_auth)
        
        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions

        return PGIndex(index_name=index_name, 
                       database=database, 
                       schema_name=schema_name,
                       host=host, 
                       port=port, 
                       username=username, 
                       password=password, 
                       dimensions=dimensions, 
                       embed_model=embed_model, 
                       enable_iam_db_auth=enable_iam_db_auth)

    index_name:str
    database:str
    schema_name:str
    host:str
    port:int
    username:str
    password:Optional[str]
    dimensions:int
    embed_model:EmbeddingType
    enable_iam_db_auth:bool=False
    initialized:bool=False

    def _get_connection(self):

        token = None

        if self.enable_iam_db_auth:
            session = boto3.Session()
            region = session.region_name
            client = session.client('rds')
            token = client.generate_db_auth_token(
                DBHostname=self.host, 
                Port=self.port, 
                DBUsername=self.username, 
                Region=region
            )
            
        password = token or self.password

        dbconn = psycopg2.connect(
            host=self.host,
            user=self.username, 
            password=password,
            port=self.port, 
            database=self.database,
            connect_timeout=30
        )

        dbconn.set_session(autocommit=True)

        if not self.initialized:

            if self.writeable:

                cur = dbconn.cursor()

                register_vector(dbconn)

                cur.execute(f'''CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.underlying_index_name()}(
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    {self.index_name}Id VARCHAR(255) unique,
                    value text,
                    metadata text,
                    embedding vector({self.dimensions})
                    );'''
                )
                cur.execute(f'CREATE INDEX IF NOT EXISTS {self.underlying_index_name()}_{self.index_name}Id_idx ON {self.schema_name}.{self.underlying_index_name()} USING hash ({self.index_name}Id);')
                cur.execute(f'CREATE INDEX IF NOT EXISTS {self.underlying_index_name()}_embedding_idx ON {self.schema_name}.{self.underlying_index_name()} USING hnsw (embedding vector_l2_ops)')
            
                cur.close()

            self.initialized = True

        return dbconn


    def add_embeddings(self, nodes:Sequence[BaseNode]) -> Sequence[BaseNode]:

        dbconn = self._get_connection()
        cur = dbconn.cursor()

        id_to_embed_map = embed_nodes(
            nodes, self.embed_model
        )
        for node in nodes:
            cur.execute(
                f'INSERT INTO {self.schema_name}.{self.underlying_index_name()} ({self.index_name}Id, value, metadata, embedding) SELECT %s, %s, %s, %s WHERE NOT EXISTS (SELECT * FROM {self.schema_name}.{self.underlying_index_name()} c WHERE c.{self.index_name}Id = %s);',
                (node.id_, node.text,  json.dumps(node.metadata), id_to_embed_map[node.id_], node.id_)
            )

        cur.close()
        dbconn.close()

        return nodes
    
    def _to_top_k_result(self, r):
        
        result = {
            'score': round(r[2], 7)
        }

        metadata = json.loads(r[1])

        if INDEX_KEY in metadata:
            index_name = metadata[INDEX_KEY]['index']
            result[index_name] = metadata[index_name]
            if 'source' in metadata:
                result['source'] = metadata['source']
        else:
            for k,v in metadata.items():
                result[k] = v
            
        return result
    
    def _to_get_embedding_result(self, r):
        
        id = r[0]
        value = r[1]
        metadata = json.loads(r[2])
        embedding = np.array(r[3]).tolist()

        result = {
            'id': id,
            'value': value,
            'embedding': embedding
        }

        for k,v in metadata.items():
            if k != INDEX_KEY:
                result[k] = v
            
        return result
    
    def top_k(self, query_bundle:QueryBundle, top_k:int=5) -> Sequence[Dict[str, Any]]:

        dbconn = self._get_connection()
        cur = dbconn.cursor()

        query_bundle = to_embedded_query(query_bundle, self.embed_model)

        cur.execute(f'''SELECT {self.index_name}Id, metadata, embedding <-> %s AS score
            FROM {self.schema_name}.{self.underlying_index_name()}
            ORDER BY score ASC LIMIT %s;''',
            (np.array(query_bundle.embedding), top_k)
        )

        results = cur.fetchall()

        top_k_results = [self._to_top_k_result(result) for result in results]

        cur.close()
        dbconn.close()

        return top_k_results

    def get_embeddings(self, ids:List[str]=[]) -> Sequence[Dict[str, Any]]:
        
        dbconn = self._get_connection()
        cur = dbconn.cursor()

        def format_ids(ids):
            return ','.join([f"'{id}'" for id in ids])
            

        cur.execute(f'''SELECT {self.index_name}Id, value, metadata, embedding
            FROM {self.schema_name}.{self.underlying_index_name()}
            WHERE {self.index_name}Id IN ({format_ids(ids)});'''
        )

        results = cur.fetchall()

        get_embeddings_results = [self._to_get_embedding_result(result) for result in results]

        cur.close()
        dbconn.close()

        return get_embeddings_results