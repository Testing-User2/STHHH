# README.md

## Overview
Minimal FastAPI bridge exposing `POST /v1/chat` for Roblox. Uses OpenAI Chat Completions. Auth via `X-Shared-Secret`. Response schema:
- `200 {"ok":true,"reply":"..."}` on success
- `200 {"ok":false,"error":"<reason>"}` on upstream failure
- `401` unauthorized
- `400` missing prompt

## Run (local)
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: SHARED_SECRET, OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)
uvicorn app:app --host 0.0.0.0 --port 8000
