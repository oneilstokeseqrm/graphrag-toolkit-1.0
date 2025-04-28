#!/bin/bash
echo "⚡ Stopping containers..."
docker compose down -v

echo "Deleting old data volumes..."
sudo rm -rf ./pgvector_data ./falkor_data

echo "Rebuilding containers..."
docker compose up --force-recreate

