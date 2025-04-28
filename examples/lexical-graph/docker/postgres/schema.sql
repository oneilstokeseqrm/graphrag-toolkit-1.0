-- Enable pgvector extension in public schema
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- Create schema for GraphRAG
CREATE SCHEMA IF NOT EXISTS graphrag;

-- Set search path to graphrag and public
SET search_path TO graphrag, public;

-- Drop and recreate the `items` table (used in some custom flows)
DROP TABLE IF EXISTS graphrag.items;
CREATE TABLE graphrag.items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    item_data JSONB,
    embedding public.vector(1024) -- changed from 1536 to 768
);

-- Drop and recreate the `chunk` table (main RAG chunks)
DROP TABLE IF EXISTS graphrag.chunk;
CREATE TABLE graphrag.chunk (
    chunkId TEXT PRIMARY KEY,
    value TEXT,
    metadata JSONB,
    embedding public.vector(1024) -- changed from 1536 to 768
);
