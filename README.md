# Roblox → LLM7.io Bridge (OpenAI SDK)

## Endpoints
- `POST /v1/chat` with `X-Shared-Secret` and `{"prompt":"..."}`  
  → `{"ok":true,"reply":"..."}` or `{"ok":false,"error":"rate|timeout|auth|http_404|http_405|upstream|net|error"}`

## Why this works
- Uses **OpenAI Async client** with `base_url=https://llm7.io/v1` (LLM7 is OpenAI-compatible).
- Avoids manual path issues that produced 405 before.

## Deploy (Render)
- Build: `pip install -r requirements.txt`  
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Env: set `SHARED_SECRET`, `LLM7_BASE=https://llm7.io/v1`, `LLM7_API_KEY=unused` (or token).

## Test
```bash
curl -s https://<your-service>/diag
curl -s -X POST https://<your-service>/v1/chat \
  -H 'X-Shared-Secret: <secret>' -H 'Content-Type: application/json' \
  -d '{"prompt":"Sag hallo"}'
