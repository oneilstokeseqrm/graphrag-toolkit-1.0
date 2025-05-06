-- Enable pgvector extension in public schema
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- Create schema for GraphRAG
CREATE SCHEMA IF NOT EXISTS graphrag;

-- Set search path to graphrag and public
SET search_path TO graphrag, public;

-- Drop and recreate the `chunk` table (main RAG chunks)
DROP TABLE IF EXISTS graphrag.chunk;
CREATE TABLE graphrag.chunk (
    chunkId TEXT PRIMARY KEY,
    value TEXT,
    metadata JSONB,
    embedding public.vector(1024)
);