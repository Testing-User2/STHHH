# app.py — DeepSeek-first bridge with explicit 402 handling and optional OpenAI fallback.
# Contract:
#   POST /v1/chat  Headers: X-Shared-Secret: <secret>
#                  Body:    {"prompt":"..."}
#   200 {"ok":true,"reply":"..."} | 200 {"ok":false,"error":"<reason>"}
#   401 for bad secret; 400 for missing_prompt

import os, time, asyncio, logging, uuid, random
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ========= ENV =========
SHARED_SECRET        = os.getenv("SHARED_SECRET", "")

# Primary: DeepSeek
DEEPSEEK_API_KEY     = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE        = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com")
DEEPSEEK_PATH        = os.getenv("DEEPSEEK_PATH", "/v1/chat/completions")  # alt: "/chat/completions"
DEEPSEEK_MODEL       = os.getenv("MODEL_NAME", "deepseek-chat")

# Optional fallback: OpenAI
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL         = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Timeouts / retries
ATTEMPT_TIMEOUT_S    = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "6.0"))
REQ_TIMEOUT_S        = float(os.getenv("REQ_TIMEOUT_SECS", "30"))
MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BACKOFF_S      = float(os.getenv("RETRY_BACKOFF_SECS", "1.0"))

# Generation
TEMP                 = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", "160"))

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
    # Map known hard failures
    if status in (401,):
        return "auth"
    if status in (402,):  # Payment Required / quota depleted at provider
        return "quota"
    # Inspect body for hints
    if body and isinstance(body, dict):
        err = body.get("error") or {}
        msg = (err.get("message") or "") if isinstance(err, dict) else ""
        code = (err.get("code") or "") if isinstance(err, dict) else ""
        low = f"{msg} {code}".lower()
        if "insufficient_quota" in low or "quota" in low or "payment" in low:
            return "quota"
        if "timeout" in low:
            return "timeout"
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

async def call_deepseek(prompt: str, deadline_ts: float) -> Tuple[bool, str, str, int]:
    if not DEEPSEEK_API_KEY:
        return False, "", "missing_key", 0

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
        "n": 1,
    }

    endpoints = [
        f"{DEEPSEEK_BASE.rstrip('/')}{DEEPSEEK_PATH}",
        f"{DEEPSEEK_BASE.rstrip('/')}/chat/completions",  # auto-fallback if PATH was wrong
    ]
    attempt_no = 0
    last_status = 0
    last_reason = "error"

    for ep in endpoints:
        attempt_no = 0
        while True:
            attempt_no += 1
            remaining = max(0.1, deadline_ts - time.time())
            this_attempt = min(ATTEMPT_TIMEOUT_S, remaining)
            try:
                status, parsed = await _attempt_post(ep, headers, body, this_attempt)
            except httpx.TimeoutException:
                status, parsed = 0, None
                last_reason = "timeout"
            except Exception:
                status, parsed = 0, None
                last_reason = "net"

            # Success path
            if status == 200 and parsed and isinstance(parsed, dict):
                try:
                    msg = (parsed.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                except Exception:
                    msg = ""
                if not msg:
                    return False, "", "empty", 200
                return True, clamp_text(msg), "", 200

            # Non-200: map reason
            if status != 0:
                last_status = status
                last_reason = reason_from_status_and_body(status, parsed)

            # Exit if not retryable or out of retries or out of time
            retryable = (last_reason in {"timeout", "net"} or (status in RETRYABLE_STATUS))
            if not retryable or attempt_no >= max(1, MAX_RETRIES) or time.time() >= deadline_ts:
                break

            # Backoff with jitter bounded by remaining time
            backoff = min(remaining, RETRY_BACKOFF_S * (2 ** (attempt_no - 1)) + random.uniform(0, 0.25))
            await asyncio.sleep(backoff)

        # try next endpoint only on clear path issues
        if last_status in (404, 405):
            continue
        break

    return False, "", last_reason, last_status

async def call_openai(prompt: str, deadline_ts: float) -> Tuple[bool, str, str, int]:
    if not OPENAI_API_KEY:
        return False, "", "missing_key", 0

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
        "n": 1,
    }

    attempt_no = 0
    while True:
        attempt_no += 1
        remaining = max(0.1, deadline_ts - time.time())
        this_attempt = min(ATTEMPT_TIMEOUT_S, remaining)
        try:
            status, parsed = await _attempt_post("https://api.openai.com/v1/chat/completions", headers, body, this_attempt)
        except httpx.TimeoutException:
            status, parsed = 0, None

        if status == 200 and parsed and isinstance(parsed, dict):
            try:
                msg = (parsed.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            except Exception:
                msg = ""
            if not msg:
                return False, "", "empty", 200
            return True, clamp_text(msg), "", 200

        reason = reason_from_status_and_body(status, parsed)
        retryable = (reason in {"timeout", "net"} or (status in RETRYABLE_STATUS))
        if not retryable or attempt_no >= max(1, MAX_RETRIES) or time.time() >= deadline_ts:
            return False, "", reason, status

        backoff = min(remaining, RETRY_BACKOFF_S * (2 ** (attempt_no - 1)) + random.uniform(0, 0.25))
        await asyncio.sleep(backoff)

# ---------- FastAPI lifecycle ----------
@app.on_event("startup")
async def _startup():
    global _http
    _http = httpx.AsyncClient(http2=True)
    log.info("startup: http client ready; deepseek_model=%s openai_model=%s", DEEPSEEK_MODEL, OPENAI_MODEL)

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
    return {"ok": True, "primary": "deepseek", "fallback_openai": bool(OPENAI_API_KEY), "model": DEEPSEEK_MODEL}

@app.api_route("/healthz", methods=["GET","HEAD"])
async def healthz():
    return {"ok": True}

@app.get("/diag")
async def diag():
    return {
        "ok": True,
        "primary": {"provider": "deepseek", "base": DEEPSEEK_BASE, "path": DEEPSEEK_PATH, "model": DEEPSEEK_MODEL, "has_key": bool(DEEPSEEK_API_KEY)},
        "fallback": {"provider": "openai", "model": OPENAI_MODEL, "has_key": bool(OPENAI_API_KEY)},
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

    # Primary DeepSeek
    ok, reply, reason, status = await call_deepseek(prompt, deadline)
    if ok:
        log.info("[chat %s] ok deepseek in %.2fs", req_id, time.time() - t0)
        return ChatOut(ok=True, reply=reply)

    log.warning("[chat %s] deepseek non-ok reason=%s status=%s", req_id, reason, status)

    # Optional fallback to OpenAI on quota/auth/endpoint issues
    if OPENAI_API_KEY and reason in {"quota", "auth", "http_404", "http_405"}:
        ok2, reply2, reason2, status2 = await call_openai(prompt, deadline)
        if ok2:
            log.info("[chat %s] ok openai fallback in %.2fs", req_id, time.time() - t0)
            return ChatOut(ok=True, reply=reply2)
        log.warning("[chat %s] openai fallback non-ok reason=%s status=%s", req_id, reason2, status2)

    # Final error envelope
    return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
