# app.py — Roblox → LLM7 bridge with robust path handling and retries.
# Requires: fastapi, uvicorn[standard], httpx
# Optional: openai (for OpenAI-compatible mode if desired)
#
# API:
#   POST /v1/chat   Header: X-Shared-Secret
#   Body: {"prompt": "..."}
#   -> 200 {"ok":true,"reply":"..."} | {"ok":false,"error":"..."} ; 401/400 on bad secret/payload

import os
import time
import uuid
import json
import asyncio
import logging
from typing import Optional, Tuple, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# ── Configuration ───────────────────────────────────────────────────────────────
# Adjustable via environment; safe operational defaults provided.

SHARED_SECRET        = os.getenv("SHARED_SECRET", "")
LLM7_API_KEY         = os.getenv("LLM7_API_KEY", "")  # token from https://token.llm7.io (Bearer)
LLM7_BASE_URL        = os.getenv("LLM7_BASE_URL", "https://llm7.io")  # no trailing slash
LLM7_MODE            = os.getenv("LLM7_MODE", "auto").lower()         # auto|openai|prompt
MODEL_NAME           = os.getenv("MODEL_NAME", "gpt-4")
SYSTEM_PROMPT        = os.getenv(
    "SYSTEM_PROMPT",
    "You are a concise Roblox NPC. Answer directly in 1–2 short sentences (9–22 words). No meta talk, no links, no code."
)

TEMP                 = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", "200"))
ATTEMPT_TIMEOUT_SECS = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "8.0"))   # per attempt
REQ_TIMEOUT_SECS     = float(os.getenv("REQ_TIMEOUT_SECS", "25"))        # whole request
MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "2"))                # beyond first attempt
HOST                 = os.getenv("HOST", "0.0.0.0")
PORT                 = int(os.getenv("PORT", "8000"))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("llm7_bridge")

# ── Models ─────────────────────────────────────────────────────────────────────
class ChatIn(BaseModel):
    prompt: str = Field(min_length=1)

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

# ── Utilities ──────────────────────────────────────────────────────────────────
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

def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]

# ── Provider Abstraction ───────────────────────────────────────────────────────
# Two wire formats:
# 1) OpenAI-compatible: POST {base}/v1/chat/completions   JSON: {model, messages, temperature, max_tokens}
# 2) Prompt endpoint:   POST {base}/v1/chat               JSON: {prompt}  -> {ok, reply|error}
#
# LLM7_MODE: "openai" (force #1), "prompt" (force #2), "auto" (try #1 then #2 on 404/405)

async def _post_openai_compatible(
    client: httpx.AsyncClient, prompt: str, attempt_timeout: float
) -> Tuple[bool, str, str, Optional[int]]:
    url = f"{LLM7_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,  # [CHANGE if your provider needs a different model id]
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
        # OpenAI-style extraction
        msg = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return False, "", "upstream", status

    if not msg:
        return False, "", "empty", status
    return True, _clip(msg, 380), "", status

async def _post_prompt_endpoint(
    client: httpx.AsyncClient, prompt: str, attempt_timeout: float
) -> Tuple[bool, str, str, Optional[int]]:
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
            # Some providers return OpenAI-like objects even on /v1/chat; be permissive.
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

    # Reuse a single client for the whole call for connection pooling
    async with httpx.AsyncClient(http2=True) as client:
        while True:
            attempt += 1
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0.05:
                return False, "", "timeout"

            attempt_timeout = min(ATTEMPT_TIMEOUT_SECS, max(0.1, remaining))

            # Select path based on mode and one-time fallback
            prefer_openai = (LLM7_MODE in ("auto", "openai")) and not used_fallback_once
            last_status: Optional[int] = None

            if prefer_openai:
                ok, reply, reason, last_status = await _post_openai_compatible(client, prompt, attempt_timeout)
                if ok:
                    log.info("[upstream openai] ok status=200")
                    return True, reply, ""
                # If 404/405 on first try in auto mode, switch to prompt endpoint
                if LLM7_MODE == "auto" and reason in {"http_404", "http_405"} and not used_fallback_once:
                    used_fallback_once = True
                    log.warning("[upstream openai] non-ok %s; trying prompt endpoint", reason)
                else:
                    log.warning("[upstream openai] non-ok reason=%s status=%s", reason, last_status)
            else:
                ok, reply, reason, last_status = await _post_prompt_endpoint(client, prompt, attempt_timeout)
                if ok:
                    log.info("[upstream prompt] ok status=200")
                    return True, reply, ""
                log.warning("[upstream prompt] non-ok reason=%s status=%s", reason, last_status)

            if attempt >= (1 + MAX_RETRIES) or not should_retry(reason):
                return False, "", reason

            backoff = min(3.0, 0.6 + 0.9 * attempt)  # bounded linear backoff
            await asyncio.sleep(backoff)

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/")
async def root():
    return {
        "ok": True,
        "provider": "llm7",
        "mode": LLM7_MODE,
        "model": MODEL_NAME,
        "base_url": LLM7_BASE_URL,
    }

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    # [CHANGE] Ensure SHARED_SECRET is set in env for production; empty means "deny all".
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
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
    else:
        log.warning("[chat %s] fail reason=%s elapsed=%.2fs", req_id, reason, elapsed)
        return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    # [CHANGE] Bind appropriately for your runtime; disable reload in production.
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
