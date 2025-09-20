import os, time, asyncio
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ---- ENV (set in Render) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
RPM            = int(os.getenv("RPM", "3"))
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
TIMEOUT_SECS   = float(os.getenv("TIMEOUT_SECS", "12"))
MIN_DELAY_SECS = float(os.getenv("MIN_DELAY_SECS", "0"))
# -----------------------------

SYSTEM_PROMPT = (
    "You are a concise conversational partner for a Roblox NPC. "
    "Mirror the user's language. Keep replies short: 1–2 sentences, 9–22 words. "
    "No links or code unless the user insists repeatedly."
)

_gap = 60.0 / max(1, RPM)
_last_call = 0.0
_gate_lock = asyncio.Lock()

async def pace():
    global _last_call
    async with _gate_lock:
        now = time.time()
        wait = _gap - (now - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.time()

class ChatIn(BaseModel):
    prompt: str

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

async def call_openai(prompt: str) -> Tuple[bool, str]:
    if not OPENAI_API_KEY:
        return False, "API key missing."
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.6,
        "max_tokens": 320,
        "n": 1,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT_SECS) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        if r.status_code == 429:
            await pace()
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        data = r.json()
    msg = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not msg:
        return False, "Empty response."
    if len(msg) > 380:
        msg = msg[:379] + "…"
    return True, msg

app = FastAPI()

@app.get("/")
async def root():
    return {"ok": True, "msg": "root alive"}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")
    t0 = time.time()
    await pace()
    ok, reply = await call_openai(prompt)
    elapsed = time.time() - t0
    if ok and elapsed < MIN_DELAY_SECS:
        await asyncio.sleep(MIN_DELAY_SECS - elapsed)
    return ChatOut(ok=ok, reply=reply if ok else None, error=None if ok else reply)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=False)
