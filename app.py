# app.py — Keyless bridge to OllamaFreeAPI (OpenAI-compatible /chat/completions)
# Contract:
#   POST /v1/chat   Headers: X-Shared-Secret: <secret>
#                   Body:    {"prompt":"..."}
#   200 {"ok":true,"reply":"..."} | 200 {"ok":false,"error":"<reason>"}
#   401 for bad secret; 400 for missing_prompt
#
# Notes:
# - No API key needed. Public gateway. Expect occasional rate limits/overload.
# - Configure one or two bases via env (OLLAMA_BASE, OLLAMA_ALT_BASE).

import os, time, asyncio, logging, uuid, random
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ========= ENV =========
SHARED_SECRET        = os.getenv("SHARED_SECRET", "")

# Primary public gateway (no auth)
OLLAMA_BASE          = os.getenv("OLLAMA_BASE", "https://ollamafreeapi.onrender.com")
OLLAMA_ALT_BASE      = os.getenv("OLLAMA_ALT_BASE", "")  # optional second base
OLLAMA_PATH          = os.getenv("OLLAMA_PATH", "/chat/completions")

MODEL_NAME           = os.getenv("MODEL_NAME", "llama3-8b")  # pick a lightweight default
TEMP                 = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", "160"))

# Timeouts / retries
ATTEMPT_TIMEOUT_S    = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "6.0"))   # per HTTP attempt
REQ_TIMEOUT_S        = float(os.getenv("REQ_TIMEOUT_SECS", "25"))        # total request budget
MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "2"))                # per-endpoint attempts
RETRY_BACKOFF_S      = float(os.getenv("RETRY_BACKOFF_SECS", "0.8"))     # base backoff

# Server
HOST                 = os.getenv("HOST", "0.0.0.0")
PORT                 = int(os.getenv("PORT", "8000"))

# ========= LOGGING =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bridge")

# ========= FASTAPI =====
app = FastAPI()
_http: Optional[httpx.AsyncClient] = None

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

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

def clamp_text(s: str, n: int = 380) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def reason_from_status_and_body(status: int, body: Optional[dict]) -> str:
    if status == 401:
        return "auth"
    if status == 404:
        return "http_404"
    if status == 405:
        return "http_405"
    if status == 429:
        return "rate"
    if status in (500, 502, 503, 504):
        return "upstream"
    if status == 0:
        return "net"
    return f"http_{status}"

async def _attempt_post(url: str, headers: dict, json: dict, timeout_s: float) -> Tuple[int, Optional[dict]]:
    assert _http is not None
    timeout = httpx.Timeout(connect=min(5.0, timeout_s), read=timeout_s, write=5.0, pool=5.0)
    r = await _http.post(url, headers=headers, json=json, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = None
    return r.status_code, body

async def call_ollama(prompt: str, deadline_ts: float) -> Tuple[bool, str, str, int]:
    """
    Returns: (ok, reply, reason, http_status)
    reason in {"timeout","net","rate","upstream","http_###","parse","empty"}
    """
    headers = {"Content-Type": "application/json"}  # no auth for public gateway
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
        "n": 1,
    }

    bases = [b for b in [OLLAMA_BASE.strip(), OLLAMA_ALT_BASE.strip()] if b]
    if not bases:
        return False, "", "http_404", 404

    last_status = 0
    last_reason = "error"

    for base in bases:
        ep = f"{base.rstrip('/')}{OLLAMA_PATH}"
        attempt_no = 0

        while True:
            attempt_no += 1
            remaining = max(0.1, deadline_ts - time.time())
            this_attempt = min(ATTEMPT_TIMEOUT_S, remaining)
            try:
                status, parsed = await _attempt_post(ep, headers, body, this_attempt)
            except httpx.TimeoutException:
                status, parsed = 0, None

            # Success path (OpenAI-style schema)
            if status == 200 and parsed and isinstance(parsed, dict):
                try:
                    msg = (
                        parsed.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                except Exception:
                    msg = ""
                if not isinstance(msg, str):
                    return False, "", "parse", status
                msg = msg.strip()
                if not msg:
                    return False, "", "empty", status
                return True, clamp_text(msg), "", 200

            # Failure
            last_status = status
            last_reason = reason_from_status_and_body(status, parsed)

            # Retry?
            retryable = (
                last_reason in {"timeout", "net", "rate", "upstream"} or
                (status in RETRYABLE_STATUS)
            )
            if not retryable or attempt_no >= max(1, MAX_RETRIES) or time.time() >= deadline_ts:
                break

            backoff = min(remaining, RETRY_BACKOFF_S * (2 ** (attempt_no - 1)) + random.uniform(0, 0.25))
            await asyncio.sleep(backoff)

        # If path is clearly wrong, try next base; otherwise stop.
        if last_status in (404, 405):
            continue
        break

    return False, "", last_reason, last_status

# ---------- FastAPI lifecycle ----------
@app.on_event("startup")
async def _startup():
    global _http
    _http = httpx.AsyncClient()  # HTTP/1.1 to avoid http2 deps
    log.info("startup: http client ready; model=%s base=%s", MODEL_NAME, OLLAMA_BASE)

@app.on_event("shutdown")
async def _shutdown():
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None
    log.info("shutdown: http client closed")

# ---------- Endpoints ----------
@app.api_route("/", methods=["GET","HEAD"])
async def root():
    return {"ok": True, "provider": "ollama-free", "model": MODEL_NAME, "base": OLLAMA_BASE}

@app.api_route("/healthz", methods=["GET","HEAD"])
async def healthz():
    return {"ok": True}

@app.get("/diag")
async def diag():
    return {
        "ok": True,
        "provider": "ollama-free",
        "model": MODEL_NAME,
        "bases": [b for b in [OLLAMA_BASE, OLLAMA_ALT_BASE] if b],
        "timeouts": {"attempt_s": ATTEMPT_TIMEOUT_S, "req_budget_s": REQ_TIMEOUT_S, "max_retries": MAX_RETRIES},
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
    deadline = t0 + REQ_TIMEOUT_S

    ok, reply, reason, status = await call_ollama(prompt, deadline)
    elapsed = time.time() - t0

    if ok:
        log.info("[chat %s] ok in %.2fs", req_id, elapsed)
        return ChatOut(ok=True, reply=reply)

    log.warning("[chat %s] non-ok reason=%s status=%s elapsed=%.2fs", req_id, reason, status, elapsed)
    # Always 200 with error envelope (except auth/missing prompt)
    return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
