import os, time, asyncio
from typing import Optional, Tuple, Dict, List
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ===== ENV =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
RPM            = int(os.getenv("RPM", "3"))
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
TIMEOUT_SECS   = float(os.getenv("TIMEOUT_SECS", "12"))
MIN_DELAY_SECS = float(os.getenv("MIN_DELAY_SECS", "0"))
MEM_TURNS      = int(os.getenv("MEMORY_TURNS", "8"))        # pairs of (user,assistant)
MEM_TTL_SECS   = int(os.getenv("MEMORY_TTL_SECS", "900"))   # 15 minutes
# ================

SYSTEM_PROMPT = (
    "You are a concise conversational partner for a Roblox NPC. "
    "Mirror the user's language. Keep replies short: 1–2 sentences, 9–22 words. "
    "No links or code unless the user insists repeatedly."
)

# ------ uniform pacing ------
_gap = 60.0 / max(1, RPM)
_last_call = 0.0
_gate_lock = asyncio.Lock()

async def pace():
    global _last_call
    async with _gate_lock:
        now = time.time()
        wait = _gap - (now - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.time()

# ------ memory store (in-process) ------
class ChatIn(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    session: Optional[str] = None
    reset: Optional[bool] = False

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

# key -> {"msgs": [ {"role": "user"/"assistant", "content": str}, ...], "t": last_ts }
MEMORY: Dict[str, Dict[str, object]] = {}

def _session_key(inp: ChatIn) -> str:
    return (inp.user_id or inp.session or "anon").strip() or "anon"

def _prune_old():
    if not MEMORY: return
    cutoff = time.time() - MEM_TTL_SECS
    stale = [k for k,v in MEMORY.items() if v.get("t", 0) < cutoff]
    for k in stale:
        MEMORY.pop(k, None)

def _append_exchange(key: str, user_text: str, assistant_text: str):
    rec = MEMORY.setdefault(key, {"msgs": [], "t": time.time()})
    msgs: List[Dict[str,str]] = rec["msgs"]  # type: ignore
    msgs.append({"role": "user", "content": user_text})
    msgs.append({"role": "assistant", "content": assistant_text})
    # clamp to last MEM_TURNS*2 messages (user+assistant per turn)
    keep = max(0, MEM_TURNS * 2)
    if keep and len(msgs) > keep:
        del msgs[:len(msgs) - keep]
    rec["t"] = time.time()

def _history_for(key: str) -> List[Dict[str, str]]:
    rec = MEMORY.get(key)
    if not rec: return []
    return list(rec["msgs"])  # shallow copy

async def call_openai(messages: List[Dict[str,str]]) -> Tuple[bool, str]:
    if not OPENAI_API_KEY:
        return False, "API key missing."
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": MODEL_NAME, "messages": messages, "temperature": 0.6, "max_tokens": 320, "n": 1}
    async with httpx.AsyncClient(timeout=TIMEOUT_SECS) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        if r.status_code == 429:
            await pace()
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        data = r.json()
    msg = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not msg:
        return False, "Empty response."
    if len(msg) > 380:
        msg = msg[:379] + "…"
    return True, msg

app = FastAPI()

@app.get("/")
async def root():
    return {"ok": True, "msg": "root alive"}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    key = _session_key(body)
    if body.reset:
        MEMORY.pop(key, None)

    # build message list: system + history + new user
    history = _history_for(key)
    messages: List[Dict[str,str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    await pace()
    ok, reply = await call_openai(messages)
    if ok:
        _append_exchange(key, prompt, reply)
        elapsed = time.time() - t0
        if elapsed < MIN_DELAY_SECS:
            await asyncio.sleep(MIN_DELAY_SECS - elapsed)
        return ChatOut(ok=True, reply=reply)
    else:
        return ChatOut(ok=False, error=reply)

@app.post("/v1/reset", response_model=ChatOut)
async def reset(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    key = _session_key(body)
    MEMORY.pop(key, None)
    return ChatOut(ok=True, reply="memory cleared")

# background prune (optional; cheap)
@app.on_event("startup")
async def _startup():
    async def loop():
        while True:
            _prune_old()
            await asyncio.sleep(60)
    asyncio.create_task(loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=False)
