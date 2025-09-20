# Roblox → OllamaFreeAPI Bridge

Minimal FastAPI server that proxies Roblox NPC prompts to **OllamaFreeAPI** (keyless, OpenAI-compatible `/chat/completions`).  
Returns compact JSON for easy Roblox consumption.

## Endpoints
- `POST /v1/chat`  
  Headers: `X-Shared-Secret: <secret>`  
  Body: `{"prompt":"..."}`
  - Success: `{"ok": true, "reply": "..."}`  
  - Error:   `{"ok": false, "error": "<reason>"}`  
- `GET /healthz` → `{"ok":true}`
- `GET /diag` → config snapshot

## Deploy on Render
1. **Create Web Service** from this repo.
2. **Environment Variables** (Render → Environment):
   - `SHARED_SECRET` — secret string (must match Roblox `Config.EXTERNAL.SHARED_SECRET`)
   - `OLLAMA_BASE=https://ollamafreeapi.onrender.com`
   - Optional: `OLLAMA_ALT_BASE` (second mirror), `MODEL_NAME`, timeouts.
3. **Build Command**: `pip install -r requirements.txt`  
   **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Deploy and check logs.

## Roblox Config
```lua
EXTERNAL = {
  URL = "https://<your-render>.onrender.com",
  PATH = "/v1/chat",
  SHARED_SECRET = "<same-as-Render>",
  TIMEOUT_SECS = 45,
}
