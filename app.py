# app.py — DeepSeek bridge (OpenAI-compatible). Real answers only. No fallbacks.
import os, time
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ===== ENV =====
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
MODEL_NAME         = os.getenv("MODEL_NAME", "deepseek-chat")   # e.g., deepseek-chat
SHARED_SECRET      = os.getenv("SHARED_SECRET", "")
ATTEMPT_TIMEOUT    = float(os.getenv("ATTEMPT_TIMEOUT_SECS", "4.0"))  # per HTTP attempt
REQ_TIMEOUT        = float(os.getenv("REQ_TIMEOUT_SECS", "30"))       # total budget
HOST               = os.getenv("HOST", "0.0.0.0")
PORT               = int(os.getenv("PORT", "8000"))
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

async def call_deepseek(prompt: str) -> Tuple[bool, str, str, int]:
    if not DEEPSEEK_API_KEY:
        return False, "", "missing_key", 0

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.5,
        "max_tokens": 160,
        "n": 1,
    }

    timeout = httpx.Timeout(
        connect=min(5.0, ATTEMPT_TIMEOUT),
        read=ATTEMPT_TIMEOUT,
        write=5.0,
        pool=5.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=body)
        except httpx.TimeoutException:
            return False, "", "timeout", 0
        except Exception:
            return False, "", "net", 0

    status = r.status_code
    if status == 200:
        try:
            data = r.json()
            msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        except Exception:
            return False, "", "parse", status
        if not msg:
            return False, "", "empty", status
        if len(msg) > 380: msg = msg[:379] + "…"
        return True, msg, "", status

    # Decode provider error details when possible
    try:
        data = r.json()
        err_msg = str(data.get("error", {}).get("message", "")) if isinstance(data, dict) else ""
        if status == 401:
            return False, "", "auth", status
        if "insufficient_quota" in err_msg.lower():
            return False, "", "quota", status
    except Exception:
        pass

    return False, "", f"http_{status}", status

app = FastAPI()

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
        "base": "https://api.deepseek.com/v1"
    }

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    t0 = time.time()
    deadline = t0 + REQ_TIMEOUT

    ok, text, reason, status = await call_deepseek(prompt)
    if ok:
        return ChatOut(ok=True, reply=text)

    print(f"[chat] status={status} reason={reason}", flush=True)
    return ChatOut(ok=False, error=reason or "error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
