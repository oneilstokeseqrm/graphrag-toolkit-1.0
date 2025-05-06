# Stop all containers (safe sweep)
docker stop $(docker ps -aq)

# Remove all containers
docker rm -f $(docker ps -aq)

# Remove your specific volumes
docker volume rm pgvector_data falkor_data

# Optional: Check for dangling volumes and nuke them
docker volume prune -f
