# app.py — DeepSeek bridge (robust, retrying, no fallbacks). FastAPI + httpx.
# Contract:
#   POST /v1/chat   Headers: X-Shared-Secret: <secret>
#                   Body:    {"prompt":"..."}
#   200 {"ok":true,"reply":"..."} | 200 {"ok":false,"error":"<reason>"}
#   401 for bad secret; 400 for missing_prompt

import os, time, asyncio, logging, uuid, random
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ========= ENV =========
DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY", "")
MODEL_NAME          = os.getenv("MODEL_NAME", "deepseek-chat")
SHARED_SECRET       = os.getenv("SHARED_SECRET", "")
ATTEMPT_TIMEOUT_S   = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "6.0"))   # per HTTP attempt
REQ_TIMEOUT_S       = float(os.getenv("REQ_TIMEOUT_SECS", "30"))        # end-to-end budget
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "2"))                # total attempts incl. first
RETRY_BACKOFF_S     = float(os.getenv("RETRY_BACKOFF_SECS", "1.0"))     # base backoff
TEMP                = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS          = int(os.getenv("MAX_TOKENS", "160"))
HOST                = os.getenv("HOST", "0.0.0.0")
PORT                = int(os.getenv("PORT", "8000"))

# ========= LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
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
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"

def reason_from_status_and_body(status: int, body: Optional[dict]) -> str:
    if status == 401:
        return "auth"
    if body and isinstance(body, dict):
        err = body.get("error") or {}
        msg = (err.get("message") or "") if isinstance(err, dict) else ""
        code = (err.get("code") or "") if isinstance(err, dict) else ""
        low = f"{msg} {code}".lower()
        if "insufficient_quota" in low or "quota" in low:
            return "quota"
        if "timeout" in low:
            return "timeout"
    return f"http_{status}"

async def call_deepseek(prompt: str, deadline_ts: float) -> Tuple[bool, str, str, int]:
    """
    Returns: (ok, reply, reason, http_status)
    reason ∈ {"auth","quota","timeout","net","parse","empty","http_###"}
    """
    if not DEEPSEEK_API_KEY:
        return False, "", "missing_key", 0

    # Create a per-call timeout object (will be updated each attempt vs deadline)
    async def attempt(timeout_s: float) -> Tuple[bool, str, str, int]:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
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
        timeout = httpx.Timeout(
            connect=min(5.0, timeout_s),
            read=timeout_s,
            write=5.0,
            pool=5.0,
        )
        assert _http is not None
        try:
            r = await _http.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers, json=body, timeout=timeout
            )
        except httpx.ReadTimeout:
            return False, "", "timeout", 0
        except httpx.ConnectTimeout:
            return False, "", "timeout", 0
        except httpx.TimeoutException:
            return False, "", "timeout", 0
        except Exception:
            return False, "", "net", 0

        status = r.status_code
        if status == 200:
            try:
                data = r.json()
            except Exception:
                return False, "", "parse", status
            msg = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not isinstance(msg, str):
                return False, "", "parse", status
            msg = msg.strip()
            if not msg:
                return False, "", "empty", status
            return True, clamp_text(msg), "", status

        # Non-200: try to parse body for better reason
        parsed = None
        try:
            parsed = r.json()
        except Exception:
            parsed = None
        return False, "", reason_from_status_and_body(status, parsed), status

    # Multi-attempt with backoff under an overall deadline
    attempt_no = 0
    while True:
        attempt_no += 1
        # compute remaining budget; keep a tiny guard
        remaining = max(0.1, deadline_ts - time.time())
        this_attempt = min(ATTEMPT_TIMEOUT_S, remaining)
        ok, text, reason, status = await attempt(this_attempt)
        if ok:
            return True, text, "", status

        # If out of budget, stop
        if time.time() >= deadline_ts:
            return False, "", "timeout", status

        # Retry only on retryable reasons/status
        retryable = (
            reason in {"timeout", "net"} or
            (isinstance(status, int) and status in RETRYABLE_STATUS)
        )
        if not retryable or attempt_no >= max(1, MAX_RETRIES):
            return False, "", reason or "error", status

        # Exponential backoff with jitter, bounded by remaining time
        backoff = min(remaining, RETRY_BACKOFF_S * (2 ** (attempt_no - 1)) + random.uniform(0, 0.25))
        await asyncio.sleep(backoff)

# ---------- FastAPI lifecycle ----------
@app.on_event("startup")
async def _startup():
    global _http
    _http = httpx.AsyncClient(http2=True)  # reuse connections; HTTP/2 helps head-of-line blocking
    log.info("startup: http client ready; model=%s", MODEL_NAME)

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
    return {"ok": True, "provider": "deepseek", "model": MODEL_NAME}

@app.api_route("/healthz", methods=["GET","HEAD"])
async def healthz():
    return {"ok": True}

@app.get("/diag")
async def diag():
    return {
        "ok": True,
        "provider": "deepseek",
        "model": MODEL_NAME,
        "has_key": bool(DEEPSEEK_API_KEY),
        "timeouts": {
            "attempt_s": ATTEMPT_TIMEOUT_S,
            "req_budget_s": REQ_TIMEOUT_S,
            "max_retries": MAX_RETRIES,
            "backoff_base_s": RETRY_BACKOFF_S,
        },
        "base": "https://api.deepseek.com/v1",
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

    ok, reply, reason, status = await call_deepseek(prompt, deadline)
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
