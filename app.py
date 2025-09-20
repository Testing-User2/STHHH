# app.py
import os, time, uuid, logging, asyncio, re
from typing import Optional, Tuple
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError

# ── Env (trim & normalize) ─────────────────────────────────────────────────────
SHARED_SECRET    = (os.getenv("SHARED_SECRET", "") or "").strip()
OPENAI_API_KEY   = (os.getenv("OPENAI_API_KEY", "") or "").strip()
OPENAI_BASE_URL  = (os.getenv("OPENAI_BASE_URL", "") or "").strip() or None  # leave empty for api.openai.com
OPENAI_ORG_ID    = (os.getenv("OPENAI_ORG_ID", "") or "").strip() or None
OPENAI_PROJECTID = (os.getenv("OPENAI_PROJECT_ID", "") or "").strip() or None

MODEL_NAME           = os.getenv("MODEL_NAME", "gpt-4o-mini")
SYSTEM_PROMPT        = os.getenv("SYSTEM_PROMPT", "You are a concise Roblox NPC. Answer directly in 1–2 short sentences (9–22 words). No meta talk, no links, no code.")
TEMP                 = float(os.getenv("TEMP", "0.6"))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", "200"))
ATTEMPT_TIMEOUT_SECS = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "8.0"))
REQ_TIMEOUT_SECS     = float(os.getenv("REQ_TIMEOUT_SECS", "25"))
MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "2"))
HOST                 = os.getenv("HOST", "0.0.0.0")
PORT                 = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("openai_bridge")

# ── Fail-fast env validation ───────────────────────────────────────────────────
def _fail(msg: str) -> None:
    log.error(msg)
    raise RuntimeError(msg)

def _validate_env() -> None:
    if not SHARED_SECRET:
        _fail("SHARED_SECRET missing")
    if not OPENAI_API_KEY:
        _fail("OPENAI_API_KEY missing")
    # Accept modern OpenAI key formats: sk-..., sk-proj-..., sk-live-...
    if not OPENAI_API_KEY.startswith("sk-"):
        _fail("OPENAI_API_KEY invalid format (must start with 'sk-')")
    # Project-scoped keys need a project header or they 401.
    if OPENAI_API_KEY.startswith("sk-proj-") and not OPENAI_PROJECTID:
        _fail("Project-scoped key detected (sk-proj-...) but OPENAI_PROJECT_ID not set")
    # If using Azure/proxy, require base_url; otherwise default must be empty.
    if OPENAI_BASE_URL and not OPENAI_BASE_URL.startswith("http"):
        _fail("OPENAI_BASE_URL must be a full URL or be empty")
_validate_env()

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECTID,
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatIn(BaseModel):
    prompt: str = Field(min_length=1)

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]

# ── OpenAI call with rich error logging ───────────────────────────────────────
def _reason_from_exc(e: Exception) -> str:
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

async def _openai_call(prompt: str) -> Tuple[bool, str, str]:
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
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            body_text = ""
            try:
                # openai-python exposes .response with .text; fall back to str(e)
                body_text = getattr(e.response, "text", "") or str(e)
            except Exception:
                body_text = str(e)
            log.error("[upstream openai] status=%s body=%s", status, body_text)
            reason = _reason_from_exc(e)
            if attempt >= (1 + MAX_RETRIES) or reason not in {"rate", "timeout", "network"}:
                return False, "", reason
            await asyncio.sleep(0.6 * attempt)
        except Exception as e:
            reason = _reason_from_exc(e)
            log.warning("[upstream openai] non-ok reason=%s", reason)
            if attempt >= (1 + MAX_RETRIES) or reason not in {"rate", "timeout", "network"}:
                return False, "", reason
            await asyncio.sleep(0.6 * attempt)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()

# Render issues HEAD / — return 200 to avoid 405 noise
@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/")
async def root():
    return {
        "ok": True,
        "provider": "openai",
        "model": MODEL_NAME,
        "base_url": OPENAI_BASE_URL or "https://api.openai.com/v1",
        "org": bool(OPENAI_ORG_ID),
        "project": bool(OPENAI_PROJECTID),
    }

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/diag")
async def diag(x_shared_secret: str = Header(default="")):
    if x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    return {
        "ok": True,
        "key_prefix": OPENAI_API_KEY[:7],
        "key_suffix": OPENAI_API_KEY[-4:],
        "base_url": OPENAI_BASE_URL or "https://api.openai.com/v1",
        "model": MODEL_NAME,
        "org": OPENAI_ORG_ID or "",
        "project": OPENAI_PROJECTID or "",
    }

@app.post("/selftest", response_model=ChatOut)
async def selftest(x_shared_secret: str = Header(default="")):
    if x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    ok, reply, reason = await _openai_call("Say OK in one sentence.")
    return ChatOut(ok=ok, reply=(reply if ok else None), error=(None if ok else reason or "error"))

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    req_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    ok, reply, reason = await _openai_call(prompt)
    elapsed = time.time() - t0

    if ok:
        log.info("[chat %s] ok in %.2fs", req_id, elapsed)
        return ChatOut(ok=True, reply=reply)
    log.warning("[chat %s] fail reason=%s elapsed=%.2fs", req_id, reason, elapsed)
    return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
