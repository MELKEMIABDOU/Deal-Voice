# Docker deployment

DealVoice runs as three internal services: the FastAPI application, Qdrant and
Ollama. Caddy is an optional public HTTPS entry point. Persistent data is kept
in named Docker volumes, not inside disposable containers.

## Files

- `Dockerfile`: builds the Python runtime and starts one API worker.
- `.dockerignore`: excludes secrets, local data, caches and reports.
- `docker-compose.yml`: runs DealVoice, Qdrant and Ollama privately.
- `docker-compose.gpu.yml`: enables NVIDIA GPU access and GPU Whisper settings.
- `docker-compose.prod.yml`: adds the Caddy HTTPS reverse proxy.
- `Caddyfile`: configures TLS, compression and pilot basic authentication.
- `.env.docker.example`: documents configuration without real secrets.

## Local CPU validation

```powershell
Copy-Item .env.docker.example .env.docker
# Add HF_TOKEN to .env.docker when audio diarisation is required.
docker compose --env-file .env.docker config
docker compose --env-file .env.docker build app
docker compose --env-file .env.docker up -d
docker compose ps
docker compose logs -f app
```

Open `http://127.0.0.1:8000`. Stop without deleting persistent data:

```powershell
docker compose down
```

Do not add `-v` unless the index, models and uploaded data may be deleted.

### Faster local inference with an AMD GPU

Docker Desktop on Windows does not provide the same AMD GPU path used by
native Ollama. Run Ollama on Windows with `OLLAMA_HOST=0.0.0.0:11434`, set
`DEALVOICE_OLLAMA_BASE_URL=http://host.docker.internal:11434` in
`.env.docker`, and recreate only the application container. Keep the port
blocked from untrusted networks with Windows Firewall.

## NVIDIA VM

Install the NVIDIA driver and NVIDIA Container Toolkit. Confirm that
`docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
works, then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --env-file .env.docker up -d --build
```

Validate WhisperX on the target VM before client use: CUDA and CTranslate2
compatibility depends on the target driver.

## Public client pilot

Point a DNS `A` record to the VM. Generate a password hash:

```bash
docker run --rm caddy:2.10.2-alpine \
  caddy hash-password --plaintext 'CHANGE-ME'
```

Put the domain, username and hash in `.env.docker`, then start:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.prod.yml --env-file .env.docker up -d --build
```

Only ports 80 and 443 should be public. Port 8000 is bound to VM localhost;
Qdrant and Ollama are internal only. Basic auth is appropriate for a limited
pilot, not final multi-user identity and tenant isolation.

## Backup

Preserve `qdrant-data`, `parent-docstore`, `raw-data` and `processed-data`
together. Restoring only Qdrant leaves child vectors without their complete
parent passages.
