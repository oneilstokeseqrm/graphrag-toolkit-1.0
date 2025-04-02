[[Home](./)]

## Amazon OpenSearch Serverless as a Vector Store

### Topics

  - [Overview](#overview)
  - [Creating an OpenSearch Serverless vector store](#creating-a-neptune-analytics-vector-store)

### Overview

You can use an Amazon OpenSearch Serverless collection as a vector store.

### Creating an OpenSearch Serverless vector store

Use the `VectorStoreFactory.for_vector_store()` static factory method to create an instance of an Amazon OpenSearch Serverless vector store.

To create an Amazon OpenSearch Serverless vector store, supply a connection string that begins `aoss://`, followed by the https endpoint of the OpenSearch Serverless collection:

```python
from graphrag_toolkit.storage import VectorStoreFactory

opensearch_connection_info = 'aoss://https://123456789012.us-east-1.aoss.amazonaws.com'

vector_store = VectorStoreFactory.for_vector_store(opensearch_connection_info)
```
