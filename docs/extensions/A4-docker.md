# A4 — Docker + docker-compose

## Goal
`docker-compose up` spins the FastAPI service + persists ChromaDB to a
volume. Anyone can run the project without setting up Python / venvs.

## Why
- **CV portfolio**: "containerised service deployable to any Docker
  host". Signals infra competence to hiring managers.
- **Reproducibility for examiner**: if the Warwick marker can't get the
  Python env working, a Docker-based fallback is useful.
- **Report — limitations section**: crosses off "deployment" from the
  gap list.

**Not a grade-lifter on its own.** Combined with A2 (FastAPI) it becomes
a legitimate story.

## Risk: ZERO
Files live at repo root (`Dockerfile`, `docker-compose.yml`). No code
changes. If Docker build fails, Python install still works.

## Scope
**New files**:
- `Dockerfile` — multi-stage: slim Python base, pip install, copy src
- `docker-compose.yml` — service + volume for chroma_db
- `.dockerignore` — exclude .venv, .git, data/html_cache, chroma_db
- `docs/extensions/running-with-docker.md` — short README

## Steps
1. Write `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY pyproject.toml .
   RUN pip install --no-cache-dir -e ".[service]"
   COPY src/ src/
   COPY service/ service/
   COPY chroma_db/ chroma_db/   # or mount as volume
   CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. Write `docker-compose.yml`:
   ```yaml
   services:
     api:
       build: .
       ports: ["8000:8000"]
       env_file: .env
       volumes:
         - ./chroma_db:/app/chroma_db
   ```
3. `.dockerignore` to keep the image small.
4. `docker-compose up --build` → confirm `curl http://localhost:8000/health` → `{"status":"ok"}`
5. Screenshot for demo log.

## Test plan
- Image builds without error (`docker-compose build`).
- Container starts and `/health` responds.
- Container serves a `/query` call end-to-end (hits ChromaDB via volume).

## Rollback
Delete `Dockerfile`, `docker-compose.yml`, `.dockerignore`. Everything
else untouched.

## Effort: 1 hour

## Depends on: A2 (FastAPI service)

## Branch: `feat/docker`
