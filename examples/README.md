## Lexical Graph Examples

### Notebooks

  - [**00-Setup**](./notebooks/00-Setup.ipynb) – Installs the lexical-graph package and additional dependencies.
  - [**01-Combined Extract and Build**](./notebooks/01-Combined-Extract-and-Build.ipynb) – An example of [performing continuous ingest](../../docs/lexical-graph/indexing.md#continous-ingest) using the `LexicalGraphIndex.extract_and_build()` method.
  - [**03-Querying**](./notebooks/04-Querying.ipynb) – Examples of [querying the graph](../../docs/lexical-graph/querying.md) using the `LexicalGraphQueryEngine` with `SemanticGuidedRetriever`.
  
## Environment Setup

The notebooks rely on `GRAPH_STORE` and `VECTOR_STORE` environment variables being properly set. These variables define where and how the graph store and vector store connect.

To set up your local environment:

1. Clone the repository and navigate to your working directory.
2. Run:

```bash
./build.sh
```

This will start and configure the following services in Docker:

- **FalkorDB** for graph storage
- **FalkorDB Browser** (accessible on `localhost:8092`) for interactive graph exploration
- **PostgreSQL with pgvector** for vector embeddings

The Postgres container auto-applies the following schema on initialization via `./postgres/schema.sql`:

```sql
-- Enable pgvector extension in public schema
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- Enable pg_trgm extension in public schema
CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public;

-- Create schema for GraphRAG
CREATE SCHEMA IF NOT EXISTS graphrag;
```

These extensions are necessary for similarity search and fuzzy matching in GraphRAG.

## AWS Foundation Model Access (Optional)

If you intend to run the CloudFormation templates instead of using Docker:

- Ensure your AWS account has access to the following Amazon Bedrock foundation models:
  - `anthropic.claude-3-5-sonnet-20240620-v1:0`
  - `cohere.embed-english-v3`

Enable model access via the [Bedrock model access console](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html).

You must deploy to an AWS region where these models are available.

## Optional: CloudFormation Stacks

If you want to deploy infrastructure in AWS, CloudFormation templates are available:

- `graphrag-toolkit-neptune-db-opensearch-serverless.json`
- `graphrag-toolkit-neptune-db-aurora-postgres.json`

These templates create:

- A Neptune serverless DB cluster
- Either OpenSearch Serverless or Aurora PostgreSQL
- A SageMaker notebook instance
- IAM roles with optional policies via the `IamPolicyArn` parameter
- An optional `ExampleNotebooksURL` parameter to auto-load the examples

> ⚠️ AWS charges apply for cloud resources.

---

Use this guide if you prefer to develop and test locally before migrating to AWS-based deployments.