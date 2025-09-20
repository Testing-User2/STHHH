# app.py — Roblox → LLM7.io bridge via OpenAI-compatible SDK (fixed exception imports)
# Contract:
#   POST /v1/chat with header X-Shared-Secret and body {"prompt":"..."}
#   → 200 {"ok":true,"reply":"..."} or {"ok":false,"error":"<key>"}
#   401 if bad secret; 400 if missing_prompt

import os, time, asyncio, logging, uuid, random
from typing import Optional, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Use only PUBLIC exceptions from the OpenAI SDK — no _private modules.
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError

# ===== ENV =====
SHARED_SECRET   = os.getenv("SHARED_SECRET", "")
LLM7_BASE       = os.getenv("LLM7_BASE", "https://llm7.io/v1")   # keep /v1
LLM7_API_KEY    = os.getenv("LLM7_API_KEY", "unused")            # or token.llm7.io token
MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-4")
TEMP            = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "160"))

ATTEMPT_TIMEOUT = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "8.0"))   # per attempt
REQ_TIMEOUT     = float(os.getenv("REQ_TIMEOUT_SECS", "25"))        # total budget
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "2"))
BACKOFF_BASE    = float(os.getenv("RETRY_BACKOFF_SECS", "0.8"))

HOST            = os.getenv("HOST", "0.0.0.0")
PORT            = int(os.getenv("PORT", "8000"))

# ===== LOG =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bridge")

# ===== APP/CLIENT =====
app = FastAPI()
_client: Optional[AsyncOpenAI] = None

class ChatIn(BaseModel):
    prompt: str

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

SYSTEM_PROMPT = (
    "You are a concise Roblox NPC. Answer directly. 1–2 sentences, 9–22 words. "
    "No meta talk, no apologies, no links, no code."
)

def clamp(s: str, n: int = 380) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def reason_from_exc(exc: Exception) -> str:
    # Order matters: more specific first.
    if isinstance(exc, RateLimitError):     return "rate"
    if isinstance(exc, APITimeoutError):    return "timeout"
    if isinstance(exc, APIConnectionError): return "net"
    if isinstance(exc, APIStatusError):
        sc = getattr(exc, "status_code", None)
        if sc == 401: return "auth"
        if sc == 404: return "http_404"
        if sc == 405: return "http_405"
        if sc == 429: return "rate"
        if sc in (500, 502, 503, 504): return "upstream"
        if isinstance(sc, int): return f"http_{sc}"
        return "upstream"
    # Generic timeout detection fallback
    txt = str(exc).lower()
    if "timeout" in txt or "timed out" in txt:
        return "timeout"
    return "error"

async def call_llm7(prompt: str, deadline_ts: float) -> Tuple[bool, str, str]:
    assert _client is not None
    attempt = 0
    last_reason = "error"

    while True:
        attempt += 1
        remaining = max(0.1, deadline_ts - time.time())
        if remaining <= 0.1:
            return False, "", "timeout"

        try:
            # SDK-level per-attempt timeout
            resp = await _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMP,
                max_tokens=MAX_TOKENS,
                timeout=ATTEMPT_TIMEOUT,
            )
            msg = (resp.choices[0].message.content or "").strip()
            if not msg:
                return False, "", "empty"
            return True, clamp(msg), ""
        except Exception as exc:
            last_reason = reason_from_exc(exc)

        retryable = last_reason in {"rate", "timeout", "net", "upstream"}
        if attempt >= max(1, MAX_RETRIES) or not retryable:
            return False, "", last_reason

        backoff = min(remaining, BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.25))
        await asyncio.sleep(backoff)

# ===== LIFECYCLE =====
@app.on_event("startup")
async def _startup():
    global _client
    _client = AsyncOpenAI(base_url=LLM7_BASE, api_key=LLM7_API_KEY)
    log.info("startup: OpenAI Async client ready; base=%s model=%s", LLM7_BASE, MODEL_NAME)

@app.on_event("shutdown")
async def _shutdown():
    log.info("shutdown: done")

# ===== ENDPOINTS =====
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"ok": True, "provider": "llm7", "base": LLM7_BASE, "model": MODEL_NAME}

@app.api_route("/healthz", methods=["GET", "HEAD"])
async def healthz():
    return {"ok": True}

@app.get("/diag")
async def diag():
    return {
        "ok": True,
        "provider": "llm7",
        "base": LLM7_BASE,
        "model": MODEL_NAME,
        "timeouts": {"attempt_s": ATTEMPT_TIMEOUT, "req_budget_s": REQ_TIMEOUT, "max_retries": MAX_RETRIES},
    }

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    req_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    deadline = t0 + REQ_TIMEOUT

    ok, reply, reason = await call_llm7(prompt, deadline)
    elapsed = time.time() - t0

    if ok:
        logging.info("[chat %s] ok in %.2fs", req_id, elapsed)
        return ChatOut(ok=True, reply=reply)

    logging.warning("[chat %s] non-ok reason=%s elapsed=%.2fs", req_id, reason, elapsed)
    return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
