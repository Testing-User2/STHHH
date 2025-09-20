# Roblox AI API (Render-ready)

FastAPI service for Roblox NPC chat. Uniform pacing to avoid rate limits. Optional Discord worker.

## Deploy on Render

1. **Fork this repo** to your GitHub.
2. In Render, **New → Web Service → Build from GitHub**. Pick this repo.
3. Render detects `render.yaml`. Confirm.
4. In the **Environment** tab of the `roblox-ai-api` service, set:
   - `OPENAI_API_KEY` = your key
   - `SHARED_SECRET`  = a strong random string (both sides must match)
   - Adjust `RPM`, `TIMEOUT_SECS`, `MIN_DELAY_SECS` as needed
5. Deploy. After it’s live, note the URL, e.g. `https://roblox-ai-api.onrender.com`.

**Allowed HTTP Domains (Roblox Studio)**  
Game Settings → Security → Allowed HTTP Domains → add your Render domain:
