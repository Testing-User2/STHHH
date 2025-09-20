# app.py — Roblox → LLM7.io bridge (chat first, completions fallback)
# POST /v1/chat  (Header: X-Shared-Secret)  Body: {"prompt":"..."}
# → 200 {"ok":true,"reply":"..."} | {"ok":false,"error":"<key>"} ; 401/400 for bad secret/payload.

import os, time, asyncio, logging, uuid, random
from typing import Optional, Tuple, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Public OpenAI SDK (no private imports)
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError
import httpx

# ================= ENV =================
SHARED_SECRET   = os.getenv("SHARED_SECRET", "")
LLM7_BASE       = os.getenv("LLM7_BASE", "https://llm7.io/v1")  # keep /v1
LLM7_API_KEY    = os.getenv("LLM7_API_KEY", "unused")           # or token.llm7.io token

MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-4")
TEMP            = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "160"))

ATTEMPT_TIMEOUT = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "8.0"))   # per attempt
REQ_TIMEOUT     = float(os.getenv("REQ_TIMEOUT_SECS", "25"))        # total budget
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "2"))
BACKOFF_BASE    = float(os.getenv("RETRY_BACKOFF_SECS", "0.8"))

# Force completions path if upstream rejects chat
COMPAT_FORCE_COMPLETIONS = os.getenv("COMPAT_FORCE_COMPLETIONS", "false").lower() in {"1","true","yes"}

HOST            = os.getenv("HOST", "0.0.0.0")
PORT            = int(os.getenv("PORT", "8000"))

# ================= LOG =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bridge")

# ================= APP/CLIENTS =================
app = FastAPI()
_client: Optional[AsyncOpenAI] = None
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

def clamp(s: str, n: int = 380) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def reason_from_exc(exc: Exception) -> str:
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
    txt = str(exc).lower()
    if "timeout" in txt or "timed out" in txt: return "timeout"
    return "error"

def _base_without_trailing_slash() -> str:
    return LLM7_BASE.rstrip("/")

def _completion_urls() -> List[str]:
    base = _base_without_trailing_slash()
    # If base already ends with /v1, prefer /completions under it; also try adding /v1 defensively.
    if base.endswith("/v1"):
        return [f"{base}/completions", f"{base}/v1/completions"]
    else:
        return [f"{base}/v1/completions", f"{base}/completions"]

async def _fallback_completions(prompt: str, deadline_ts: float) -> Tuple[bool, str, str]:
    """Try legacy /completions endpoint with a concatenated prompt."""
    assert _http is not None
    headers = {
        "Authorization": f"Bearer {LLM7_API_KEY}",
        "Content-Type": "application/json",
    }
    # Compose a simple instruct-style prompt
    full_prompt = (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"User: {prompt}\n\n"
        f"Assistant:"
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
        "n": 1,
    }

    last_reason, last_status = "error", 0
    for url in _completion_urls():
        attempt = 0
        while True:
            attempt += 1
            remaining = max(0.1, deadline_ts - time.time())
            if remaining <= 0.1:
                return False, "", "timeout"
            timeout = httpx.Timeout(connect=min(5.0, remaining), read=min(ATTEMPT_TIMEOUT, remaining),
                                    write=5.0, pool=5.0)
            try:
                r = await _http.post(url, headers=headers, json=payload, timeout=timeout)
                status = r.status_code
                data: Optional[Dict[str, Any]]
                try:
                    data = r.json()
                except Exception:
                    data = None

                if status == 200 and isinstance(data, dict):
                    # Accept either completions or chat-like structure
                    text = ""
                    if isinstance(data.get("choices"), list) and data["choices"]:
                        choice = data["choices"][0]
                        if isinstance(choice, dict):
                            if "text" in choice and isinstance(choice["text"], str):
                                text = choice["text"].strip()
                            elif "message" in choice and isinstance(choice["message"], dict):
                                text = (choice["message"].get("content") or "").strip()
                    if text:
                        return True, clamp(text), ""
                    return False, "", "empty"

                # Map status
                if status == 401: last_reason = "auth"
                elif status == 404: last_reason = "http_404"
                elif status == 405: last_reason = "http_405"
                elif status == 429: last_reason = "rate"
                elif status in (500, 502, 503, 504): last_reason = "upstream"
                else: last_reason = f"http_{status}"
                last_status = status

            except httpx.TimeoutException:
                last_reason, last_status = "timeout", 0
            except httpx.HTTPError:
                last_reason, last_status = "net", 0

            # Retry on transient
            transient = last_reason in {"timeout", "rate", "upstream", "net"}
            if attempt >= max(1, MAX_RETRIES) or not transient:
                break
            backoff = min(remaining, BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.25))
            await asyncio.sleep(backoff)

        # Try next URL variant
        if last_reason in {"http_404", "http_405"}:
            continue
        break

    return False, "", last_reason

async def call_llm7(prompt: str, deadline_ts: float) -> Tuple[bool, str, str]:
    """Try chat first (OpenAI SDK). On 405 or forced compat, fall back to /completions."""
    # Either forced completions or attempt chat first
    if COMPAT_FORCE_COMPLETIONS:
        return await _fallback_completions(prompt, deadline_ts)

    # Try chat.completions via SDK
    attempt, last_reason = 0, "error"
    while True:
        attempt += 1
        remaining = max(0.1, deadline_ts - time.time())
        if remaining <= 0.1:
            return False, "", "timeout"

        try:
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

        # If upstream rejects method/path, switch to completions immediately
        if last_reason == "http_405":
            return await _fallback_completions(prompt, deadline_ts)

        transient = last_reason in {"timeout", "rate", "upstream", "net"}
        if attempt >= max(1, MAX_RETRIES) or not transient:
            break
        backoff = min(remaining, BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.25))
        await asyncio.sleep(backoff)

    # Final fallback if we didn't try it yet and reason hints router/path issues
    if last_reason in {"http_404", "http_405"}:
        return await _fallback_completions(prompt, deadline_ts)
    return False, "", last_reason

# ================= LIFECYCLE =================
@app.on_event("startup")
async def _startup():
    global _client, _http
    _client = AsyncOpenAI(base_url=LLM7_BASE.rstrip("/"), api_key=LLM7_API_KEY)
    _http = httpx.AsyncClient()
    log.info("startup: clients ready; base=%s model=%s compat_force_completions=%s",
             LLM7_BASE, MODEL_NAME, COMPAT_FORCE_COMPLETIONS)

@app.on_event("shutdown")
async def _shutdown():
    global _http
    if _http:
        await _http.aclose()
        _http = None
    log.info("shutdown: http client closed")

# ================= ENDPOINTS =================
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "ok": True,
        "provider": "llm7",
        "base": LLM7_BASE,
        "model": MODEL_NAME,
        "compat_force_completions": COMPAT_FORCE_COMPLETIONS,
    }

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
        "compat_force_completions": COMPAT_FORCE_COMPLETIONS,
        "completion_urls": _completion_urls(),
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
