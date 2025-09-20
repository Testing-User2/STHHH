# app.py — Roblox → LLM7 bridge with HTTP/2 auto, path fallback, retries.
# Deps: fastapi, uvicorn[standard], httpx
# Optional: h2  # [CHANGE] Install to enable HTTP/2 if desired.

import os
import time
import uuid
import asyncio
import logging
from typing import Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# ── Config ─────────────────────────────────────────────────────────────────────
SHARED_SECRET        = os.getenv("SHARED_SECRET", "")               # [CHANGE] set non-empty
LLM7_API_KEY         = os.getenv("LLM7_API_KEY", "")                # [CHANGE] bearer token if required
LLM7_BASE_URL        = os.getenv("LLM7_BASE_URL", "https://llm7.io")
LLM7_MODE            = os.getenv("LLM7_MODE", "auto").lower()       # auto|openai|prompt
MODEL_NAME           = os.getenv("MODEL_NAME", "gpt-4")             # [CHANGE] model id for your provider
SYSTEM_PROMPT        = os.getenv(
    "SYSTEM_PROMPT",
    "You are a concise Roblox NPC. Answer directly in 1–2 short sentences (9–22 words). No meta talk, no links, no code."
)
TEMP                 = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", "200"))
ATTEMPT_TIMEOUT_SECS = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "8.0"))
REQ_TIMEOUT_SECS     = float(os.getenv("REQ_TIMEOUT_SECS", "25"))
MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "2"))
HOST                 = os.getenv("HOST", "0.0.0.0")
PORT                 = int(os.getenv("PORT", "8000"))
HTTP2_ENV            = os.getenv("HTTP2", "auto").lower()           # auto|true|false

def _detect_http2_enabled() -> bool:
    if HTTP2_ENV in ("false", "0", "no", "off"):
        return False
    if HTTP2_ENV in ("true", "1", "yes", "on"):
        try:
            import h2  # noqa: F401
            return True
        except Exception:
            logging.getLogger("llm7_bridge").warning("HTTP/2 requested but 'h2' not installed; using HTTP/1.1")
            return False
    try:
        import h2  # noqa: F401
        return True
    except Exception:
        return False

HTTP2_ENABLED = _detect_http2_enabled()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm7_bridge")

# ── Schemas ────────────────────────────────────────────────────────────────────
class ChatIn(BaseModel):
    prompt: str = Field(min_length=1)

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

# ── Helpers ────────────────────────────────────────────────────────────────────
def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]

def reason_from_status(status: Optional[int]) -> str:
    if status is None:
        return "error"
    if status == 401:
        return "auth"
    if status == 404:
        return "http_404"
    if status == 405:
        return "http_405"
    if status == 429:
        return "rate"
    return f"http_{status}"

def should_retry(reason: str) -> bool:
    return reason in {"rate", "timeout", "network", "upstream"}

# ── Upstream calls ─────────────────────────────────────────────────────────────
async def _post_openai_compatible(client: httpx.AsyncClient, prompt: str, attempt_timeout: float) -> Tuple[bool, str, str, Optional[int]]:
    url = f"{LLM7_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,  # [CHANGE] provider-specific model id if needed
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
    }
    headers = {}
    if LLM7_API_KEY:
        headers["Authorization"] = f"Bearer {LLM7_API_KEY}"
    try:
        resp = await client.post(url, headers=headers, json=payload, timeout=attempt_timeout)
    except httpx.ReadTimeout:
        return False, "", "timeout", None
    except httpx.ConnectTimeout:
        return False, "", "timeout", None
    except httpx.TransportError:
        return False, "", "network", None

    status = resp.status_code
    if status != 200:
        return False, "", reason_from_status(status), status
    try:
        data = resp.json()
        msg = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return False, "", "upstream", status
    if not msg:
        return False, "", "empty", status
    return True, _clip(msg, 380), "", status

async def _post_prompt_endpoint(client: httpx.AsyncClient, prompt: str, attempt_timeout: float) -> Tuple[bool, str, str, Optional[int]]:
    url = f"{LLM7_BASE_URL.rstrip('/')}/v1/chat"
    payload = {"prompt": prompt}
    headers = {}
    if LLM7_API_KEY:
        headers["Authorization"] = f"Bearer {LLM7_API_KEY}"
    try:
        resp = await client.post(url, headers=headers, json=payload, timeout=attempt_timeout)
    except httpx.ReadTimeout:
        return False, "", "timeout", None
    except httpx.ConnectTimeout:
        return False, "", "timeout", None
    except httpx.TransportError:
        return False, "", "network", None

    status = resp.status_code
    if status == 401:
        return False, "", "auth", status
    if status in (400, 404, 405, 429):
        return False, "", reason_from_status(status), status
    if status != 200:
        return False, "", f"http_{status}", status

    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("ok") is True:
            msg = (data.get("reply") or "").strip()
        else:
            msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    except Exception:
        return False, "", "upstream", status
    if not msg:
        return False, "", "empty", status
    return True, _clip(msg, 380), "", status

async def call_llm7(prompt: str) -> Tuple[bool, str, str]:
    deadline = time.time() + REQ_TIMEOUT_SECS
    attempt = 0
    used_fallback_once = False
    async with httpx.AsyncClient(http2=HTTP2_ENABLED) as client:
        while True:
            attempt += 1
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0.05:
                return False, "", "timeout"
            attempt_timeout = min(ATTEMPT_TIMEOUT_SECS, max(0.1, remaining))

            prefer_openai = (LLM7_MODE in ("auto", "openai")) and not used_fallback_once
            if prefer_openai:
                ok, reply, reason, status = await _post_openai_compatible(client, prompt, attempt_timeout)
                if ok:
                    log.info("[upstream openai] ok")
                    return True, reply, ""
                if LLM7_MODE == "auto" and reason in {"http_404", "http_405"} and not used_fallback_once:
                    used_fallback_once = True
                    log.warning("[upstream openai] %s; trying prompt endpoint", reason)
                else:
                    log.warning("[upstream openai] non-ok reason=%s status=%s", reason, status)
            else:
                ok, reply, reason, status = await _post_prompt_endpoint(client, prompt, attempt_timeout)
                if ok:
                    log.info("[upstream prompt] ok")
                    return True, reply, ""
                log.warning("[upstream prompt] non-ok reason=%s status=%s", reason, status)

            if attempt >= (1 + MAX_RETRIES) or not should_retry(reason):
                return False, "", reason
            await asyncio.sleep(min(3.0, 0.6 + 0.9 * attempt))

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/")
async def root():
    return {
        "ok": True,
        "provider": "llm7",
        "mode": LLM7_MODE,
        "model": MODEL_NAME,
        "base_url": LLM7_BASE_URL,
        "http2": HTTP2_ENABLED,
    }

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:  # [CHANGE] ensure set in env
        raise HTTPException(status_code=401, detail="unauthorized")
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    req_id = uuid.uuid4().hex[:8]
    start = time.time()
    ok, reply, reason = await call_llm7(prompt)
    elapsed = time.time() - start

    if ok:
        log.info("[chat %s] ok in %.2fs", req_id, elapsed)
        return ChatOut(ok=True, reply=reply)
    log.warning("[chat %s] fail reason=%s elapsed=%.2fs", req_id, reason, elapsed)
    return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
