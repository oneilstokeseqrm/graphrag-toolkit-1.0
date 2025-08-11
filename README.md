## GraphRAG Toolkit

The graphrag-toolkit is a collection of Python tools for building graph-enhanced Generative AI applications.

> **Note**: This is a dummy change to test PR creation functionality.

> **1 August 2025** Release 3.11.0 includes a Neo4j graph store for the lexical graph and performance improvements to the traversal-based retriever. FalkorDB support has been moved into a lexical-graph-contrib package.

> **4 June 2025** Release 3.8.0 includes a separate BYOKG-RAG package, which allows users to bring their own knowledge graph and perform complex question answering over it.

> **28 May 2025** Release 3.7.0 includes an MCP server that dynamically generates tools and tool descriptions (one per tenant in a multi-tenant graph).

Installation instructions and requirements are detailed separately with each tool.

### Lexical Graph

The [lexical-graph](./lexical-graph/) provides a framework for automating the construction of a [hierarchical lexical graph](./docs/lexical-graph/graph-model.md) from unstructured data, and composing question-answering strategies that query this graph when answering user questions.

### BYOKG-RAG

[BYOKG-RAG](./byokg-rag/) is a novel approach to Knowledge Graph Question Answering (KGQA) that combines the power of Large Language Models (LLMs) with structured knowledge graphs. The system allows users to bring their own knowledge graph and perform complex question answering over it.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

