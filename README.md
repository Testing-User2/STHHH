# Roblox → LLM7.io Bridge

Minimal FastAPI service that proxies Roblox NPC prompts to **LLM7.io** (OpenAI-compatible).
Returns a compact JSON envelope tailored for Roblox.

## Endpoints
- `POST /v1/chat`
  - Headers: `X-Shared-Secret: <secret>`
  - Body: `{"prompt":"..."}`
  - Success: `{"ok": true, "reply": "..."}`  
  - Error:   `{"ok": false, "error": "<reason>"}`  (`rate`, `timeout`, `upstream`, `http_404`, etc.)
- `GET /healthz` → `{"ok": true}`
- `GET /diag` → config snapshot (base, paths tried, model)

## Deploy (Render)
1. Create a Web Service from this repo.
2. **Environment Variables**:
   - `SHARED_SECRET=<your secret>`  (must match Roblox `Config.EXTERNAL.SHARED_SECRET`)
   - `LLM7_BASE=https://llm7.io/v1`
   - `LLM7_API_KEY=unused` (or your free token from https://token.llm7.io)
   - Optional tuning: `MODEL_NAME`, `TEMP`, `MAX_TOKENS`, timeouts.
3. **Build Command**: `pip install -r requirements.txt`  
   **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Deploy.

## Quick Test
```bash
curl -s https://<your-render>/diag

curl -s -X POST https://<your-render>/v1/chat \
  -H 'X-Shared-Secret: <your secret>' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Sag hallo"}'
