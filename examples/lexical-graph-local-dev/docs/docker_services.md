# Docker Services Overview for GraphRAG Deployment

This document describes the services defined in the `docker-compose.yml` file used for setting up a GraphRAG environment. It includes containerized services for FalkorDB, a FalkorDB browser UI, and a PostgreSQL database with the `pgvector` extension enabled.

---

## Services

### 1. `falkordb`
- **Image**: `falkordb/falkordb:latest`
- **Description**: Runs the FalkorDB graph database, which uses Redis as its backend.
- **Ports**:
  - `6379`: Redis/FalkorDB main port.
  - `3000`: Optional REST API for FalkorDB if exposed.
- **Volume**: Persists graph data using `falkor_data`.
- **Network**: Connected to `graphrag_network`.

### 2. `falkordb-browser`
- **Image**: `falkordb/falkordb-browser:latest`
- **Description**: Provides a web-based interface for interacting with FalkorDB.
- **Ports**:
  - `8092:8080`: Web UI exposed on localhost:8092.
- **Environment Variables**:
  - `FALKORDB_BROWSER_REDIS_HOST`: Hostname of the FalkorDB service.
  - `FALKORDB_BROWSER_REDIS_PORT`: Port for Redis.
  - `FALKORDB_BROWSER_REDIS_USE_TLS`: TLS setting for secure Redis communication (disabled in this setup).
- **Depends On**: `falkordb`
- **Network**: Connected to `graphrag_network`.

### 3. `postgres`
- **Image**: `pgvector/pgvector:0.6.2-pg16`
- **Description**: PostgreSQL 16 image with the `pgvector` extension pre-installed for vector search capabilities.
- **Ports**:
  - `5432`: PostgreSQL default port.
- **Environment Variables**:
  - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: Injected from environment or `.env` file.
- **Volumes**:
  - `pgvector_data`: Data persistence.
  - `./postgres/schema.sql`: Initializes the database schema.
- **Network**: Connected to `graphrag_network`.

---

## `schema.sql`

This SQL file is used to bootstrap the PostgreSQL container with necessary extensions and a custom schema:

```sql
-- Enable pgvector extension in public schema
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- Enable pg_trgm extension in public schema
CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public;

-- Create schema for GraphRAG
CREATE SCHEMA IF NOT EXISTS graphrag;
```

These extensions are required for vector similarity search and trigram-based indexing within the GraphRAG framework.

---

## Networks

- **graphrag_network**: A dedicated Docker bridge network for inter-container communication.

---

## Volumes

- `falkor_data`: Persists FalkorDB graph state.
- `pgvector_data`: Persists PostgreSQL data including vector embeddings and schema definitions.
