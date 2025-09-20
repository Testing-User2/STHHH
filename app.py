# app.py – Render bridge for Roblox to LLM7.io (OpenAI-compatible)
# POST /v1/chat  (Header: X-Shared-Secret)  Body: {"prompt":"..."}
# -> 200 {"ok":true,"reply":"..."} | {"ok":false,"error":"..."} ; 401/400 on bad secret/payload.

import os
import time
import uuid
import logging
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError

# ── Environment ──────────────────────────────────────────
SHARED_SECRET   = os.getenv("SHARED_SECRET", "")
LLM7_API_KEY    = os.getenv("LLM7_API_KEY", "unused")   # use token from https://token.llm7.io or 'unused'
MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-4")
TEMP            = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "200"))
ATTEMPT_TIMEOUT = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "8.0"))
REQ_TIMEOUT     = float(os.getenv("REQ_TIMEOUT_SECS", "25"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "2"))
HOST            = os.getenv("HOST", "0.0.0.0")
PORT            = int(os.getenv("PORT", "8000"))

SYSTEM_PROMPT = (
    "You are a concise Roblox NPC. Answer directly in 1–2 short sentences (9–22 words). "
    "No meta talk, no links, no code."
)

# ── FastAPI & Clients ─────────────────────────────────────
app = FastAPI()
client = AsyncOpenAI(base_url="https://llm7.io/v1", api_key=LLM7_API_KEY)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm7")

class ChatIn(BaseModel):
    prompt: str

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

def reason_from_exc(e: Exception) -> str:
    if isinstance(e, RateLimitError):     return "rate"
    if isinstance(e, APITimeoutError):    return "timeout"
    if isinstance(e, APIConnectionError): return "network"
    if isinstance(e, APIStatusError):
        sc = getattr(e, "status_code", None)
        if sc == 401: return "auth"
        if sc == 404: return "http_404"
        if sc == 405: return "http_405"
        if sc == 429: return "rate"
        return f"http_{sc}"
    if "timeout" in str(e).lower(): return "timeout"
    return "error"

async def call_llm7(prompt: str) -> tuple[bool, str, str]:
    """Call LLM7 with retry logic, returning (ok, reply, reason)."""
    attempt = 0
    deadline = time.time() + REQ_TIMEOUT
    while True:
        attempt += 1
        remaining = max(0.1, deadline - time.time())
        if remaining <= 0.1:
            return False, "", "timeout"
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMP,
                max_tokens=MAX_TOKENS,
                timeout=ATTEMPT_TIMEOUT,
            )
            msg = (resp.choices[0].message.content or "").strip()
            if not msg:
                return False, "", "empty"
            return True, msg[:380], ""
        except Exception as e:
            reason = reason_from_exc(e)
            log.warning("[chat attempt=%d] non-ok reason=%s", attempt, reason)
            if attempt >= MAX_RETRIES or reason not in {"rate","timeout","network","upstream"}:
                return False, "", reason
            await asyncio.sleep(min(3.0, 0.8 * attempt))

@app.get("/")
async def root():
    return {"ok": True, "provider": "llm7", "model": MODEL_NAME}

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

    req_id = uuid.uuid4().hex[:8]
    start = time.time()
    ok, reply, reason = await call_llm7(prompt)
    elapsed = time.time() - start

    if ok:
        log.info("[chat %s] ok in %.2fs", req_id, elapsed)
        return ChatOut(ok=True, reply=reply)
    else:
        log.warning("[chat %s] fail reason=%s elapsed=%.2fs", req_id, reason, elapsed)
        return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
