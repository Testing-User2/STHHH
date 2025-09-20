# app.py — FastAPI, no queue, deadline-aware OpenAI call with robust retries
import os, time, asyncio, random
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ===== ENV (set in Render) =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
# End-to-end budget per request (seconds). Pick 90–150.
REQ_TIMEOUT    = float(os.getenv("REQ_TIMEOUT_SECS", "120"))
# Per attempt HTTP timeout. Keep < REQ_TIMEOUT.
OPENAI_ATTEMPT_TIMEOUT = float(os.getenv("OPENAI_ATTEMPT_TIMEOUT_SECS", "30"))
HOST           = os.getenv("HOST", "0.0.0.0")
PORT           = int(os.getenv("PORT", "8000"))
# ===============================

SYSTEM_PROMPT = (
    "You are a concise conversational partner for a Roblox NPC. "
    "Mirror the user's language. Keep replies short: 1–2 sentences, 9–22 words. "
    "No links or code unless the user insists repeatedly."
)

class ChatIn(BaseModel):
    prompt: str

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

def _retry_after_secs(r: httpx.Response) -> Optional[float]:
    ra = r.headers.get("retry-after")
    if ra:
        try:
            v = float(ra)
            if v > 0: return v
        except: pass
    try:
        j = r.json()
        if isinstance(j, dict):
            msg = str(j.get("error", {}).get("message", ""))
            # crude extractor like "try again in 10s"
            import re
            m = re.search(r"(\d+)\s*s", msg)
            if m: return float(m.group(1))
    except: pass
    return None

async def call_openai_with_deadline(prompt: str, deadline: float) -> Tuple[bool, str]:
    if not OPENAI_API_KEY:
        return False, "missing_key"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.6,
        "max_tokens": 320,
        "n": 1,
    }

    # set granular timeouts on the client to avoid hanging a whole request
    timeout = httpx.Timeout(
        connect=min(10.0, OPENAI_ATTEMPT_TIMEOUT),
        read=OPENAI_ATTEMPT_TIMEOUT,
        write=10.0,
        pool=10.0,
    )

    backoff = 1.5
    attempts = 0
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            remaining = deadline - time.time()
            if remaining <= 1.0:
                return False, "timeout"

            try:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            except httpx.TimeoutException:
                attempts += 1
                # try again if we still have time
                if deadline - time.time() > 3.0:
                    await asyncio.sleep(min(backoff, 8.0))
                    backoff = min(backoff * 1.7, 12.0)
                    continue
                return False, "timeout"

            if r.status_code == 200:
                data = r.json()
                msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                if not msg:
                    return False, "empty"
                if len(msg) > 380:
                    msg = msg[:379] + "…"
                return True, msg

            # explicit quota surface
            try:
                data = r.json()
                err_msg = str(data.get("error", {}).get("message", ""))
                if "insufficient_quota" in err_msg.lower():
                    return False, "quota"
            except Exception:
                pass

            # handle transient / rate errors with backoff, respecting deadline
            if r.status_code in (429, 500, 502, 503, 504):
                ra = _retry_after_secs(r)
                sleep = ra if ra is not None else min(2.0 + attempts * 2.0, 12.0)
                attempts += 1
                remaining = deadline - time.time()
                if remaining <= sleep + 1.0:
                    return False, "timeout"
                await asyncio.sleep(sleep + random.random() * 0.3)
                continue

            return False, f"http_{r.status_code}"

app = FastAPI()

@app.api_route("/", methods=["GET","HEAD"])
async def root():
    return {"ok": True, "msg": "alive"}

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

    deadline = time.time() + REQ_TIMEOUT
    ok, out = await call_openai_with_deadline(prompt, deadline)
    if ok:
        return ChatOut(ok=True, reply=out)
    # compact log for diagnosis
    print(f"[chat] non-ok -> {out}", flush=True)
    return ChatOut(ok=False, error=out)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
