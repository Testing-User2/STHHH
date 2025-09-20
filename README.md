# README.md

## Overview
Minimal HTTPS bridge for Roblox → DeepSeek (OpenAI-compatible). No memory. No fallbacks.
Endpoint contract:
- `POST /v1/chat` with header `X-Shared-Secret: <secret>` and JSON `{"prompt":"..."}`
- Returns `200 {"ok":true,"reply":"..."}` or `200 {"ok":false,"error":"..."}`.  
- Returns `401` only for bad secret.

## Deploy (Render)
1. Create a new **Web Service** from this repo.
2. Set environment variables:
   - `DEEPSEEK_API_KEY` = your key
   - `MODEL_NAME` = `deepseek-chat` (default)
   - `SHARED_SECRET` = shared secret (must match Roblox Config)
   - `ATTEMPT_TIMEOUT_SECS` = `4.0` (tune if needed)
   - `REQ_TIMEOUT_SECS` = `30`
   - `HOST` = `0.0.0.0`
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Verify
- `GET /diag` → `{"ok":true,"provider":"deepseek","model":"deepseek-chat","has_key":true,"base":"https://api.deepseek.com/v1"}`
- `POST /v1/chat` with header `X-Shared-Secret` and body `{"prompt":"hi"}` → returns a short model reply.

## Roblox Wiring
- In your `Config.EXTERNAL`:
  - `URL = "https://<your-render>.onrender.com"`
  - `PATH = "/v1/chat"`
  - `SHARED_SECRET = "<same-as-Render>"`
- Ensure **Allow HTTP Requests** is enabled in your experience settings.

## Tuning
- If timeouts: raise `ATTEMPT_TIMEOUT_SECS` to 5.0–6.0.
- If rate errors from provider: throttle on client side; avoid spam calls.
