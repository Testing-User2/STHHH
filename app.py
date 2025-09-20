# app.py
import os, time, uuid, logging
from typing import Optional, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError

SHARED_SECRET        = os.getenv("SHARED_SECRET", "")  # [CHANGE]
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "") # [CHANGE]
OPENAI_BASE_URL      = os.getenv("OPENAI_BASE_URL", "").strip() or None  # [CHANGE] only for Azure/proxy

MODEL_NAME           = os.getenv("MODEL_NAME", "gpt-4o-mini")  # [CHANGE]
SYSTEM_PROMPT        = os.getenv("SYSTEM_PROMPT", "You are a concise Roblox NPC. Answer directly in 1–2 short sentences (9–22 words). No meta talk, no links, no code.")  # [CHANGE if needed]
TEMP                 = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", "200"))
ATTEMPT_TIMEOUT_SECS = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "8.0"))
REQ_TIMEOUT_SECS     = float(os.getenv("REQ_TIMEOUT_SECS", "25"))
MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "2"))
HOST                 = os.getenv("HOST", "0.0.0.0")
PORT                 = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("openai_bridge")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else AsyncOpenAI(api_key=OPENAI_API_KEY)

class ChatIn(BaseModel):
    prompt: str = Field(min_length=1)

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]

def reason_from_exc(e: Exception) -> str:
    if isinstance(e, RateLimitError):     return "rate"
    if isinstance(e, APITimeoutError):    return "timeout"
    if isinstance(e, APIConnectionError): return "network"
    if isinstance(e, APIStatusError):
        sc = getattr(e, "status_code", None)
        if sc == 401: return "auth"
        if isinstance(sc, int): return f"http_{sc}"
        return "upstream"
    if "timeout" in str(e).lower(): return "timeout"
    return "error"

async def call_openai(prompt: str) -> Tuple[bool, str, str]:
    deadline = time.time() + REQ_TIMEOUT_SECS
    attempt = 0
    while True:
        attempt += 1
        remaining = max(0.0, deadline - time.time())
        if remaining <= 0.05:
            return False, "", "timeout"
        attempt_timeout = min(ATTEMPT_TIMEOUT_SECS, max(0.1, remaining))
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMP,
                max_tokens=MAX_TOKENS,
                timeout=attempt_timeout,
            )
            msg = (resp.choices[0].message.content or "").strip()
            if not msg:
                return False, "", "empty"
            return True, _clip(msg, 380), ""
        except Exception as e:
            reason = reason_from_exc(e)
            log.warning("[upstream openai attempt=%d] non-ok reason=%s", attempt, reason)
            if attempt >= (1 + MAX_RETRIES) or reason not in {"rate", "timeout", "network"}:
                return False, "", reason

app = FastAPI()

@app.get("/")
async def root():
    return {"ok": True, "provider": "openai", "model": MODEL_NAME, "base_url": OPENAI_BASE_URL or "https://api.openai.com/v1"}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    req_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    ok, reply, reason = await call_openai(prompt)
    elapsed = time.time() - t0

    if ok:
        log.info("[chat %s] ok in %.2fs", req_id, elapsed)
        return ChatOut(ok=True, reply=reply)
    log.warning("[chat %s] fail reason=%s elapsed=%.2fs", req_id, reason, elapsed)
    return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
