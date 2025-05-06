#!/bin/bash

echo "Stopping and removing containers, volumes, and networks..."
docker compose down -v --remove-orphans

echo "Ensuring containers are removed..."
docker rm -f falkordb falkordb-browser pgvector-db 2>/dev/null

echo "Removing named volumes..."
docker volume rm -f pgvector_data falkor_data 2>/dev/null

echo "Pruning dangling volumes (if any)..."
docker volume prune -f

echo "Clearing extracted directory..."
rm -rf extracted

echo "Rebuilding containers..."
docker compose up -d --force-recreate

echo "Reset complete."