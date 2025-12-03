# Copyright FalkorDB.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import numpy as np

from pgvector.psycopg2 import register_vector
from typing import List, Sequence, Dict, Any, Optional, Callable
from urllib.parse import urlparse, parse_qs

from graphrag_toolkit.lexical_graph.metadata import FilterConfig, type_name_for_key_value, format_datetime
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig, EmbeddingType
from graphrag_toolkit.lexical_graph.storage.vector import VectorIndex, to_embedded_query
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

from llama_index.core.schema import BaseNode, QueryBundle
from llama_index.core.indices.utils import embed_nodes
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters


logger = logging.getLogger(__name__)

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    from psycopg2.errors import UniqueViolation, UndefinedTable
except ImportError as e:
    raise ImportError(
        "psycopg2 and/or pgvector packages not found, install with 'pip install psycopg2-binary pgvector'"
    ) from e
    

def to_sql_operator(operator: FilterOperator) -> tuple[str, Callable[[Any], str]]:
    """
    Converts a filter operator into an SQL operator and its respective value formatter.

    This function maps a `FilterOperator` enum value to a tuple that contains the SQL
    representation of the operator and a callable function to format the value for use
    in SQL queries. If the provided operator is unsupported, a `ValueError` is raised.

    Args:
        operator (FilterOperator): The filter operator to be converted to its SQL
            equivalent.

    Returns:
        tuple[str, Callable[[Any], str]]: A tuple where the first element is the SQL
            operator as a string, and the second element is a callable to format
            the value for use with the SQL operator.

    Raises:
        ValueError: If the provided `operator` is not supported.
    """
    default_value_formatter = lambda x: x
    
    operator_map = {
        FilterOperator.EQ: ('=', default_value_formatter), 
        FilterOperator.GT: ('>', default_value_formatter), 
        FilterOperator.LT: ('<', default_value_formatter), 
        FilterOperator.NE: ('<>', default_value_formatter), 
        FilterOperator.GTE: ('>=', default_value_formatter), 
        FilterOperator.LTE: ('<=', default_value_formatter), 
        #FilterOperator.IN: ('in', default_value_formatter),  # In array (string or number)
        #FilterOperator.NIN: ('nin', default_value_formatter),  # Not in array (string or number)
        #FilterOperator.ANY: ('any', default_value_formatter),  # Contains any (array of strings)
        #FilterOperator.ALL: ('all', default_value_formatter),  # Contains all (array of strings)
        FilterOperator.TEXT_MATCH: ('LIKE', lambda x: f"%%{x}%%"),
        FilterOperator.TEXT_MATCH_INSENSITIVE: ('~*', default_value_formatter),
        #FilterOperator.CONTAINS: ('contains', default_value_formatter),  # metadata array contains value (string or number)
        FilterOperator.IS_EMPTY: ('IS NULL', default_value_formatter),  # the field is not exist or empty (null or empty array)
    }

    if operator not in operator_map:
        raise ValueError(f'Unsupported filter operator: {operator}')
    
    return operator_map[operator]

def formatter_for_type(type_name:str) -> Callable[[Any], str]:
    """
    Returns a formatter function corresponding to the given type name. The formatter
    function takes a value as input and returns the value formatted appropriately
    for the specified type. The supported types include 'text', 'timestamp', 'int',
    and 'float'.

    Args:
        type_name (str): The name of the type for which the formatter function is
            required. Supported types are 'text', 'timestamp', 'int', and 'float'.

    Returns:
        Callable[[Any], str]: A function that formats values according to the
            specified type.

    Raises:
        ValueError: If the specified type name is not supported.
    """
    if type_name == 'text':
        return lambda x: f"'{x}'"
    elif type_name == 'timestamp':
        return lambda x: f"'{format_datetime(x)}'"
    elif type_name in ['int', 'float']:
        return lambda x:x
    else:
        raise ValueError(f'Unsupported type name: {type_name}')


def parse_metadata_filters_recursive(metadata_filters:MetadataFilters) -> str:
    """
    Parses metadata filters recursively into a SQL-compatible string representation.

    This function takes a `MetadataFilters` object, traverses its hierarchical
    structure, and converts it into SQL-like conditions. It handles different
    filter operations, logical conditions (AND, OR, NOT), and nested filters.

    Args:
        metadata_filters (MetadataFilters): A MetadataFilters object containing
            the hierarchical structure of filters and logical conditions to
            parse into a SQL-compatible string.

    Returns:
        str: A SQL-compatible string representation of the provided metadata
            filters.

    Raises:
        ValueError: If invalid metadata filter types are encountered within the
            `metadata_filters` structure, or if an unsupported filter condition
            is provided.
    """
    def to_key(key: str) -> str:
        """
        Convert a metadata key to the JSONB path used in queries.
        """
        return f"metadata->'source'->'metadata'->>'{key}'"
    
    def to_sql_filter(f: MetadataFilter) -> str:
        """
        Convert a single MetadataFilter into a SQL predicate string.
        """
        key = to_key(f.key)
        (operator, operator_formatter) = to_sql_operator(f.operator)

        if f.operator == FilterOperator.IS_EMPTY:
            return f"({key} {operator})"
        else:
            type_name = type_name_for_key_value(f.key, f.value)
            type_formatter = formatter_for_type(type_name)
            return f"(({key})::{type_name} {operator} {type_formatter(operator_formatter(str(f.value)))})"
    
    condition = metadata_filters.condition.value

    filter_strs = []

    for metadata_filter in metadata_filters.filters:
        if isinstance(metadata_filter, MetadataFilter):
            if metadata_filters.condition == FilterCondition.NOT:
                raise ValueError(f'Expected MetadataFilters for FilterCondition.NOT, but found MetadataFilter')
            filter_strs.append(to_sql_filter(metadata_filter))
        elif isinstance(metadata_filter, MetadataFilters):
            filter_strs.append(parse_metadata_filters_recursive(metadata_filter))
        else:
            raise ValueError(f'Invalid metadata filter type: {type(metadata_filter)}')
        
    if metadata_filters.condition == FilterCondition.NOT:
        return f"(NOT {' '.join(filter_strs)})"
    elif metadata_filters.condition in (FilterCondition.AND, FilterCondition.OR):
        condition = f' {metadata_filters.condition.value.upper()} '
        return f"({condition.join(filter_strs)})"
    else:
        raise ValueError(f'Unsupported filters condition: {metadata_filters.condition}')


def filter_config_to_sql_filters(filter_config:FilterConfig) -> str:
    """
    Converts a given FilterConfig object into an SQL WHERE clause-like filter string.
    """
    if filter_config is None or filter_config.source_filters is None:
        return ''
    return parse_metadata_filters_recursive(filter_config.source_filters)


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
                  enable_iam_db_auth:bool=False):
        """
        Creates an instance of the `PGIndex` class based on the provided database connection
        information, index name, and embedding configuration.

        **Neon / DSN note**: any additional libpq connection parameters (e.g. `sslmode`,
        `sslrootcert`, `sslcert`, `sslkey`, `options`, `channel_binding`, `application_name`)
        can be supplied as standard URI query parameters and will be forwarded to
        `psycopg2.connect(...)`. The special flag `enable_iam_db_auth` is interpreted by
        this class and is *not* forwarded to psycopg2.
        """
        def compute_enable_iam_db_auth(s: str, default: bool) -> bool:
            """
            Decide whether IAM auth should be enabled based on the query string.
            """
            if 'enable_iam_db_auth' in s.lower():
                return 'enable_iam_db_auth=true' in s.lower()
            else:
                return default
        
        parsed = urlparse(connection_string)
        query_params = parse_qs(parsed.query or "")

        # Basic connection components
        database = parsed.path[1:] if parsed.path else database
        host = parsed.hostname or host
        port = parsed.port or port
        username = parsed.username or username
        password = parsed.password or password

        # Determine IAM auth from query string (toolkit flag, not libpq)
        enable_iam_db_auth = compute_enable_iam_db_auth(parsed.query or "", enable_iam_db_auth)

        # Extract libpq-compatible connection kwargs from query parameters.
        # We intentionally do NOT pass `enable_iam_db_auth` through to psycopg2.
        allowed_connect_params = {
            "sslmode",
            "sslrootcert",
            "sslcert",
            "sslkey",
            "options",
            "channel_binding",
            "application_name",
        }
        connect_kwargs: Dict[str, Any] = {}

        for key in allowed_connect_params:
            if key in query_params and query_params[key]:
                # Use the last value if multiple are provided
                connect_kwargs[key] = query_params[key][-1]

        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions

        return PGIndex(
            index_name=index_name,
            database=database,
            schema_name=schema_name,
            host=host,
            port=port,
            username=username,
            password=password,
            dimensions=dimensions,
            embed_model=embed_model,
            enable_iam_db_auth=enable_iam_db_auth,
            connect_kwargs=connect_kwargs,
        )

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
    # Additional libpq/psycopg2 connection keyword args derived from the DSN query string.
    # For example: sslmode, sslrootcert, channel_binding, options, etc.
    connect_kwargs:Dict[str, Any] = {}

    def _get_connection(self):
        """
        Establish a connection to the database, configure it, and optionally initialize
        the schema and indexes required for the application.

        - If IAM DB auth is enabled, we obtain a temporary auth token from RDS and
          use that in place of the password.
        - In all cases, we pass any supported libpq kwargs found in the original DSN
          query string (e.g. sslmode=require) to psycopg2.connect(...).
        """
        token = None

        if self.enable_iam_db_auth:
            client = GraphRAGConfig.rds  # via __getattr__
            region = GraphRAGConfig.aws_region
            token = client.generate_db_auth_token(
                DBHostname=self.host,
                Port=self.port,
                DBUsername=self.username,
                Region=region
            )

        password = token or self.password

        # Apply libpq-compatible kwargs (e.g. sslmode, sslrootcert, channel_binding)
        connect_kwargs = dict(self.connect_kwargs or {})
        connect_kwargs.setdefault("connect_timeout", 30)

        dbconn = psycopg2.connect(
            host=self.host,
            user=self.username, 
            password=password,
            port=self.port, 
            database=self.database,
            **connect_kwargs,
        )

        dbconn.set_session(autocommit=True)

        register_vector(dbconn)

        if not self.initialized:

            cur = dbconn.cursor()

            try:

                if self.writeable:

                    try:
                        cur.execute(
                            f'''CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.underlying_index_name()}(
                                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                {self.index_name}Id VARCHAR(255) unique,
                                value text,
                                metadata jsonb,
                                embedding vector({self.dimensions})
                                );'''
                        )
                    except UniqueViolation:
                        # For alt approaches, see: https://stackoverflow.com/questions/29900845/create-schema-if-not-exists-raises-duplicate-key-error
                        logger.warning(f"Table already exists, so ignoring CREATE: {self.underlying_index_name()}")
                        pass

                    index_name = f'{self.underlying_index_name()}_{self.index_name}Id_idx'
                    try:
                        cur.execute(
                            f'CREATE INDEX IF NOT EXISTS {index_name} '
                            f'ON {self.schema_name}.{self.underlying_index_name()} '
                            f'USING hash ({self.index_name}Id);'
                        )
                    except UniqueViolation:
                        logger.warning(f"Index already exists, so ignoring CREATE: {index_name}")
                        pass

                    index_name = f'{self.underlying_index_name()}_{self.index_name}Id_embedding_idx'
                    try:
                        cur.execute(
                            f'CREATE INDEX IF NOT EXISTS {index_name} '
                            f'ON {self.schema_name}.{self.underlying_index_name()} '
                            f'USING hnsw (embedding vector_l2_ops)'
                        )
                    except UniqueViolation:
                        logger.warning(f"Index already exists, so ignoring CREATE: {index_name}")
                        pass
                    
                    index_name = f'{self.underlying_index_name()}_{self.index_name}Id_gin_idx'
                    try:
                        cur.execute(
                            f'CREATE INDEX IF NOT EXISTS {index_name} '
                            f'ON {self.schema_name}.{self.underlying_index_name()} '
                            f'USING GIN (metadata)'
                        )
                    except UniqueViolation:
                        logger.warning(f"Index already exists, so ignoring CREATE: {index_name}")
                        pass

            finally:
                cur.close()

            self.initialized = True

        return dbconn


    def add_embeddings(self, nodes:Sequence[BaseNode]) -> Sequence[BaseNode]:
        """
        Adds embeddings for a sequence of nodes into the database.
        """
        if not self.writeable:
            raise IndexError(f'Index {self.index_name} is read-only')

        dbconn = self._get_connection()
        cur = dbconn.cursor()

        try:

            id_to_embed_map = embed_nodes(
                nodes, self.embed_model
            )
            for node in nodes:
                cur.execute(
                    f'INSERT INTO {self.schema_name}.{self.underlying_index_name()} '
                    f'({self.index_name}Id, value, metadata, embedding) '
                    f'SELECT %s, %s, %s, %s '
                    f'WHERE NOT EXISTS ('
                    f'  SELECT * FROM {self.schema_name}.{self.underlying_index_name()} c '
                    f'  WHERE c.{self.index_name}Id = %s'
                    f');',
                    (node.id_, node.text, json.dumps(node.metadata),
                     id_to_embed_map[node.id_], node.id_)
                )

        except UndefinedTable as e:
            if self.tenant_id.is_default_tenant():
                raise e
            else:
                logger.warning(f'Multi-tenant index {self.underlying_index_name()} does not exist')

        finally:

            cur.close()
            dbconn.close()

        return nodes
    
    def _to_top_k_result(self, r):
        """
        Converts a raw result into a structured Top K result dictionary.
        """
        result = {
            'score': round(r[2], 7)
        }

        metadata_payload = r[1]
        if isinstance(metadata_payload, dict):
            metadata = metadata_payload
        else:
            metadata = json.loads(metadata_payload)

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
        """
        Convert the result from an embedding query into a formatted dictionary.
        """
        id = r[0]
        value = r[1]

        metadata_payload = r[2]
        if isinstance(metadata_payload, dict):
            metadata = metadata_payload
        else:
            metadata = json.loads(metadata_payload)
 
        embedding = np.array(r[3], dtype=object).tolist()

        result = {
            'id': id,
            'value': value,
            'embedding': embedding
        }

        for k,v in metadata.items():
            if k != INDEX_KEY:
                result[k] = v
            
        return result
    
    def top_k(self, query_bundle:QueryBundle, top_k:int=5, filter_config:Optional[FilterConfig]=None) -> Sequence[Dict[str, Any]]:
        """
        Retrieves the top K results from the database based on the provided query
        and optional filter configuration.
        """
        dbconn = self._get_connection()
        cur = dbconn.cursor()

        top_k_results = []

        try:

            where_clause =  filter_config_to_sql_filters(filter_config)
            where_clause = f'WHERE {where_clause}' if where_clause else ''

            logger.debug(f'filter: {where_clause}')

            query_bundle = to_embedded_query(query_bundle, self.embed_model)

            sql = f'''SELECT {self.index_name}Id, metadata, embedding <-> %s AS score
                FROM {self.schema_name}.{self.underlying_index_name()}
                {where_clause}
                ORDER BY score ASC LIMIT %s;'''
            
            logger.debug(f'sql: {sql}')
        
            cur.execute(sql, (np.array(query_bundle.embedding), top_k))

            results = cur.fetchall()

            top_k_results.extend(
                [self._to_top_k_result(result) for result in results]
            )

        except UndefinedTable as e:
            logger.warning(f'Index {self.underlying_index_name()} does not exist')

        finally:
            cur.close()
            dbconn.close()

        return top_k_results

    def get_embeddings(self, ids:List[str]=[]) -> Sequence[Dict[str, Any]]:
        """
        Retrieves embeddings from the database for the given list of IDs.
        """
        dbconn = self._get_connection()
        cur = dbconn.cursor()

        def format_ids(ids):
            return ','.join([f"'{id}'" for id in set(ids)])
        
        get_embeddings_results = []

        try:

            cur.execute(
                f'''SELECT {self.index_name}Id, value, metadata, embedding
                    FROM {self.schema_name}.{self.underlying_index_name()}
                    WHERE {self.index_name}Id IN ({format_ids(ids)});'''
            )

            results = cur.fetchall()

            get_embeddings_results.extend(
                [self._to_get_embedding_result(result) for result in results]
            )

        except UndefinedTable as e:
            logger.warning(f'Index {self.underlying_index_name()} does not exist')

        finally:
            cur.close()
            dbconn.close()

        return get_embeddings_results