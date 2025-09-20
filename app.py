# app.py — Real-model answers only. No fallbacks. Groq/OpenAI switchable.
import os, time, asyncio
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ===== ENV =====
PROVIDER       = os.getenv("PROVIDER", "groq").lower()   # "groq" | "openai"
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "llama3-8b-8192")  # groq default
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
ATTEMPT_TIMEOUT_SECS = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "3.0"))  # per HTTP attempt
REQ_TIMEOUT_SECS     = float(os.getenv("REQ_TIMEOUT_SECS", "30"))       # end-to-end cap
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
# =================

SYSTEM_PROMPT = (
    "You are a concise Roblox NPC. Answer directly. 1–2 sentences, 9–22 words. "
    "No meta talk, no apologies, no code, no links."
)

class ChatIn(BaseModel):
    prompt: str

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

def _base_url_and_key():
    if PROVIDER == "groq":
        return "https://api.groq.com/openai/v1", GROQ_API_KEY
    elif PROVIDER == "openai":
        return "https://api.openai.com/v1", OPENAI_API_KEY
    return "", ""

async def call_provider(prompt: str) -> Tuple[bool, str, str]:
    base, key = _base_url_and_key()
    if not base or not key:
        return False, "", "missing_key"

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.5,
        "max_tokens": 120,
        "n": 1,
    }

    timeout = httpx.Timeout(
        connect=min(5.0, ATTEMPT_TIMEOUT_SECS),
        read=ATTEMPT_TIMEOUT_SECS,
        write=5.0,
        pool=5.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(f"{base}/chat/completions", headers=headers, json=body)
        except httpx.TimeoutException:
            return False, "", "timeout"
        except Exception:
            return False, "", "net"

    if r.status_code == 200:
        try:
            data = r.json()
            msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        except Exception:
            return False, "", "parse"
        if not msg:
            return False, "", "empty"
        if len(msg) > 380:
            msg = msg[:379] + "…"
        return True, msg, ""
    # map common errors
    try:
        data = r.json()
        err_msg = str(data.get("error", {}).get("message", ""))
        if "insufficient_quota" in err_msg.lower():
            return False, "", "quota"
        if r.status_code == 401:
            return False, "", "auth"
    except Exception:
        pass
    return False, "", f"http_{r.status_code}"

app = FastAPI()

@app.api_route("/", methods=["GET","HEAD"])
async def root():
    # No secrets, just to verify provider/model at runtime.
    return {"ok": True, "provider": PROVIDER, "model": MODEL_NAME}

@app.api_route("/healthz", methods=["GET","HEAD"])
async def healthz():
    return {"ok": True}

@app.api_route("/diag", methods=["GET"])
async def diag():
    # Helps confirm Render is configured to hit the intended provider; no key values are exposed.
    base, key = _base_url_and_key()
    return {
        "ok": True,
        "provider": PROVIDER,
        "model": MODEL_NAME,
        "has_key": bool(key),
        "base": base,
    }

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    t0 = time.time()
    deadline = t0 + REQ_TIMEOUT_SECS

    # Single real attempt; return only model text or a concrete error
    ok, text, reason = await call_provider(prompt)
    if ok:
        return ChatOut(ok=True, reply=text)

    # Log compactly for inspection on Render
    print(f"[chat] provider={PROVIDER} non-ok -> {reason}", flush=True)

    # Return error → Roblox shows short German error mapped in ChatAIService
    return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
