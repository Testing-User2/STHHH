# Roblox AI Bridge (stateless, paced)

Single-endpoint FastAPI service that calls OpenAI with strict pacing and retries. Designed for Roblox NPCs: one prompt in → one reply out. No memory. Includes robust 429/5xx handling and a small server-side queue to smooth spikes.

## Endpoints

- `POST /v1/chat`  
  Body: `{"prompt": "text"}`  
  Headers: `X-Shared-Secret: <your secret>`  
  Response: `{"ok": true, "reply": "..."}` or `{"ok": false, "error": "busy|timeout|quota|http_###"}`

- `GET /healthz` – health check

## Environment Variables

See `.env.example`. On Render, set these in the Environment tab (or use `render.yaml`). Required:
- `OPENAI_API_KEY`
- `SHARED_SECRET`

Recommended:
- `MODEL_NAME=gpt-4o-mini`
- `RPM=2`
- `TIMEOUT_SECS=18`
- `REQ_TIMEOUT_SECS=45`
- `QUEUE_SIZE=128`

## Local Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill values
export $(grep -v '^#' .env | xargs)  # or use a dotenv tool
uvicorn app:app --host 0.0.0.0 --port 8000
