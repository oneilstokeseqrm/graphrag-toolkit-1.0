# build.sh

This script is used to build and start the containers for a new deployment of the application using Docker Compose. It is intended for **initial deployments** or **redeployments** without resetting volumes, removing data, or clearing persistent state.

## Usage

```bash
chmod +x build.sh
./build.sh
```

## What it does

- Executes `docker compose up -d --build` to:
  - Build the Docker images using the `Dockerfile`s defined in the project.
  - Start the services in detached mode (`-d`) so the terminal remains available.
  - Automatically pull required images if not already present.
  - Rebuild containers if source code has changed.

## Important Notes

- This script does **not** remove any existing containers, volumes, or data.
- It is safe to run on top of an existing deployment if you are deploying an updated version of your app.
- Make sure your `.env` and `docker-compose.yml` files are configured properly before running the script.

## Related Scripts

- See [`reset.sh`](reset.md) for a full environment reset, including data deletion and volume pruning.
