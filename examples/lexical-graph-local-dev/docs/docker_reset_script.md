# Docker Environment Reset Script

This script is used to **fully reset a local Docker-based development environment** for graphrag-toolkit. The script will reset FalkorDB, PGVector, and optionally other components. It performs cleanup of containers, networks, volumes, and extracted data, followed by a fresh container rebuild.

## Filename

Use `reset.sh` (file is located in lexical-graph-contrib/docker) and run it with:

```bash
bash reset.sh
```

> **Note:** Make sure the script is executable (`chmod +x reset.sh`) or invoke it with `bash`.

---

## Script Breakdown

```bash
#!/bin/bash
```
- Standard shebang to run the script using `bash`.

---

### 1. Stop and Remove Docker Resources

```bash
echo "Stopping and removing containers, volumes, and networks..."
docker compose down -v --remove-orphans
```

- **`docker compose down`** stops and removes containers defined in `docker-compose.yml`.
- **`-v`** removes associated anonymous volumes.
- **`--remove-orphans`** removes containers not defined in the current Compose file but part of the same project network.

---

### 2. Explicitly Remove Named Containers

```bash
echo "Ensuring containers are removed..."
docker rm -f falkordb falkordb-browser pgvector-db 2>/dev/null
```

- Forcefully removes specific named containers, if they still exist.
- Errors are suppressed using `2>/dev/null`.

---

### 3. Remove Named Volumes

```bash
echo "Removing named volumes..."
docker volume rm -f pgvector_data falkor_data 2>/dev/null
```

- Deletes project-specific Docker volumes that might persist after shutdown.

---

### 4. Prune Dangling Volumes

```bash
echo "Pruning dangling volumes (if any)..."
docker volume prune -f
```

- Removes **dangling (unused)** Docker volumes that may be left behind.

---

### 5. Delete Local Directories

```bash
echo "Clearing extracted directory..."
rm -rf extracted
```

- Cleans up the local `./extracted` directory used to store intermediate files (like parsed documents, indexes, or temp outputs).

---

### 6. Rebuild and Start Containers

```bash
echo "Rebuilding containers..."
docker compose up -d --force-recreate
```

- **`-d`** runs containers in detached mode.
- **`--force-recreate`** ensures all containers are recreated even if configuration hasn't changed.

---

### 7. Final Message

```bash
echo "Reset complete."
```

- Indicates successful completion of the reset process.

---

## Use Cases

- Full environment reset between development sessions
- Clean-up after corrupt container or volume states
- Ensures a consistent baseline environment for troubleshooting or testing

---

## Warnings

- **Data Loss**: This script removes all persistent data and should not be used on production environments.
- **Rebuild Time**: Fresh container creation may take time depending on image sizes and network speed.

