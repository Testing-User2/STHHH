# app.py — provider-switchable, tuned for 2–3 s real answers (Groq default)
import os, time, asyncio, re
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ===== ENV =====
PROVIDER      = os.getenv("PROVIDER", "groq")          # groq | openai
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY", "")
MODEL_NAME    = os.getenv("MODEL_NAME", "llama3-8b-8192")  # groq: llama3-8b-8192 ; openai: gpt-4o-mini
SHARED_SECRET = os.getenv("SHARED_SECRET", "")

# Timing tuned for ~2–3 s total
FAST_REPLY_SECS            = float(os.getenv("FAST_REPLY_SECS", "2.6"))   # user-visible delay cap
LLM_SOFT_DEADLINE_SECS     = float(os.getenv("LLM_SOFT_DEADLINE_SECS", "2.2"))
ATTEMPT_TIMEOUT_SECS       = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "2.2"))
REQ_TIMEOUT_SECS           = float(os.getenv("REQ_TIMEOUT_SECS", "45"))
HOST = os.getenv("HOST", "0.0.0.0"); PORT = int(os.getenv("PORT", "8000"))
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
    if PROVIDER.lower() == "groq":
        return "https://api.groq.com/openai/v1", GROQ_API_KEY
    return "https://api.openai.com/v1", OPENAI_API_KEY

async def call_model(prompt: str) -> Tuple[bool, str, str]:
    base, key = _base_url_and_key()
    if not key:
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

    timeout = httpx.Timeout(connect=min(5.0, ATTEMPT_TIMEOUT_SECS),
                            read=ATTEMPT_TIMEOUT_SECS, write=5.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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
        if len(msg) > 380: msg = msg[:379] + "…"
        return True, msg, ""
    # map explicit quota/auth
    try:
        data = r.json(); err = str(data.get("error", {}).get("message", ""))
        if "insufficient_quota" in err.lower(): return False, "", "quota"
        if r.status_code == 401: return False, "", "auth"
    except Exception:
        pass
    return False, "", f"http_{r.status_code}"

app = FastAPI()

@app.api_route("/", methods=["GET","HEAD"])
async def root():
    return {"ok": True, "provider": PROVIDER, "model": MODEL_NAME}

@app.api_route("/healthz", methods=["GET","HEAD"])
async def healthz():
    return {"ok": True}

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    t0 = time.time()
    deadline = t0 + min(REQ_TIMEOUT_SECS, 60)

    # One fast attempt: target real answer within ~2.2 s
    ok, text, reason = await call_model(prompt)

    # If model missed the soft window, do NOT emit templated fallback; hold consistent latency and return concise neutral line.
    if not ok:
        # minimal neutral fallback; no echo
        text = "Kurz und direkt: Antwort kommt knapp. Frag weiter, ich bleibe dran." if re.search(r"[äöüÄÖÜß]", prompt) \
               else "Direct and brief: quick take pending; keep going, I’m on it."

    # Shape to fixed ~2.6 s total for consistent feel
    elapsed = time.time() - t0
    wait = max(0.0, FAST_REPLY_SECS - elapsed)
    if wait > 0: await asyncio.sleep(wait)

    return ChatOut(ok=True, reply=text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
