[[Home](./)]

## FalkorDB as a Graph Store

### Topics

  - [Overview](#overview)
  - [Registering FalkorDB as a graph store](#registering-falkordb-as-a-graph-store)
  - [Creating a FalkorDB graph store](#creating-a-falkordb-graph-store)

### Overview

You can use FalkorDB as a graph store.

### Registering FalkorDB as a graph store

Before creating a FalkorDB graph store, you must register the `FalkorDBGraphStoreFactory` with the `GraphStoreFactory`:

```python
from graphrag_toolkit.storage import GraphStoreFactory
from graphrag_toolkit.store.graph.falkordb_graph_store import FalkorDBGraphStoreFactory

GraphStoreFactory.register_graph_store_factory_method(FalkorDBGraphStoreFactory)

```

### Creating a FalkorDB graph store

You can use the `GraphStoreFactory.for_graph_store()` static factory method to create an instance of a FalkorDB graph store.

The FalkorDB graph store currently supports the [SemanticGuidedRetriever](./querying.md#semanticguidedretriever). It does not support the [TraversalBasedRetriever](./querying.md#traversalbasedretriever).

To create a [FalkorDB Cloud](https://app.falkordb.cloud/) graph store, supply a connection string that begins `falkordb://`, followed by the FalkorDB endpoint:

```python
from graphrag_toolkit.storage import GraphStoreFactory
from graphrag_toolkit.store.graph.falkordb_graph_store import FalkorDBGraphStoreFactory

falkordb_connection_info = 'falkordb://your-falkordb-endpoint'

GraphStoreFactory.register_graph_store_factory_method(FalkorDBGraphStoreFactory)

graph_store = GraphStoreFactory.for_graph_store(falkordb_connection_info)
```

You may also need to pass a username and password, and specify whether or not to use SSL:

```python
from graphrag_toolkit.storage import GraphStoreFactory

falkordb_connection_info = 'falkordb://<your-falkordb-endpoint>'

graph_store = GraphStoreFactory.for_graph_store(
    falkordb_connection_info,
    username='<username>',
    password='<password>',
    ssl=True
)
```

To create a local FalkorDB graph store, supply a connection string that has only `falkordb://`;

```python
from graphrag_toolkit.storage import GraphStoreFactory

falkordb_connection_info = 'falkordb://'

graph_store = GraphStoreFactory.for_graph_store(falkordb_connection_info)
```

