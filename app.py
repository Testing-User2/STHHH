import os, time, asyncio, random
from typing import Optional, Tuple, Dict, List
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ========= ENV =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
RPM            = int(os.getenv("RPM", "3"))               # sustained requests/min
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
TIMEOUT_SECS   = float(os.getenv("TIMEOUT_SECS", "12"))   # per HTTP call timeout
MIN_DELAY_SECS = float(os.getenv("MIN_DELAY_SECS", "0"))  # optional floor latency

MEM_TURNS      = int(os.getenv("MEMORY_TURNS", "8"))      # last N user/assistant pairs
MEM_TTL_SECS   = int(os.getenv("MEMORY_TTL_SECS", "900")) # 15 min TTL

QUEUE_SIZE     = int(os.getenv("QUEUE_SIZE", "64"))       # server-side backlog
REQ_TIMEOUT    = float(os.getenv("REQ_TIMEOUT_SECS", "25")) # end-to-end cap
# =======================

SYSTEM_PROMPT = (
    "You are a concise conversational partner for a Roblox NPC. "
    "Mirror the user's language. Keep replies short: 1–2 sentences, 9–22 words. "
    "No links or code unless the user insists repeatedly."
)

# ---- pacing (uniform gap) ----
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

# ---- memory store (in-process) ----
class ChatIn(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    session: Optional[str] = None
    reset: Optional[bool] = False

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

MEMORY: Dict[str, Dict[str, object]] = {}
def _key(inp: ChatIn) -> str:
    return (inp.user_id or inp.session or "anon").strip() or "anon"

def _hist_get(k: str) -> List[Dict[str,str]]:
    rec = MEMORY.get(k); 
    if not rec: return []
    return list(rec["msgs"])  # type: ignore

def _hist_append(k: str, user_text: str, assistant_text: str):
    rec = MEMORY.setdefault(k, {"msgs": [], "t": time.time()})
    msgs: List[Dict[str,str]] = rec["msgs"]  # type: ignore
    msgs.append({"role":"user","content":user_text})
    msgs.append({"role":"assistant","content":assistant_text})
    keep = max(0, MEM_TURNS*2)
    if keep and len(msgs) > keep:
        del msgs[:len(msgs)-keep]
    rec["t"] = time.time()

async def _prune_loop():
    while True:
        if MEMORY:
            cutoff = time.time() - MEM_TTL_SECS
            for k in [k for k,v in MEMORY.items() if v.get("t",0) < cutoff]:
                MEMORY.pop(k, None)
        await asyncio.sleep(60)

# ---- OpenAI call with robust retry ----
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
            m = str(j.get("error",{}).get("message",""))
            # crude parse like: "try again in 5s"
            import re
            z = re.search(r"(\d+)\s*s", m)
            if z: return float(z.group(1))
    except: pass
    return None

async def call_openai(messages: List[Dict[str,str]]) -> Tuple[bool, str]:
    if not OPENAI_API_KEY:
        return False, "missing_key"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": MODEL_NAME, "messages": messages, "temperature": 0.6, "max_tokens": 320, "n": 1}

    async with httpx.AsyncClient(timeout=TIMEOUT_SECS) as client:
        attempts = 0
        while attempts < 3:
            attempts += 1
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            if r.status_code == 200:
                data = r.json()
                msg = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip()
                if not msg: return False, "empty"
                return True, msg if len(msg) <= 380 else msg[:379]+"…"

            # insufficient quota
            try:
                data = r.json()
                err_msg = str(data.get("error",{}).get("message",""))
                if "insufficient_quota" in err_msg or "quota" in err_msg.lower():
                    return False, "quota"
            except: pass

            # 429/5xx backoff
            if r.status_code in (429, 500, 502, 503, 504):
                ra = _retry_after_secs(r)
                backoff = ra if ra is not None else min(2 + attempts*3, 20)
                await asyncio.sleep(backoff + random.random()*0.3)
                await pace()
                continue

            return False, f"http_{r.status_code}"

        return False, "retry_exhausted"

# ---- global request queue (single worker) ----
class Job:
    __slots__ = ("messages","key","user_text","fut")
    def __init__(self, messages, key, user_text, fut):
        self.messages = messages
        self.key = key
        self.user_text = user_text
        self.fut = fut

REQUEST_Q: asyncio.Queue[Job] = asyncio.Queue(maxsize=QUEUE_SIZE)

async def worker_loop():
    while True:
        job = await REQUEST_Q.get()
        try:
            await pace()
            ok, r = await call_openai(job.messages)
            if ok:
                _hist_append(job.key, job.user_text, r)
                job.fut.set_result(ChatOut(ok=True, reply=r))
            else:
                # map errors to stable strings
                if r == "quota":
                    job.fut.set_result(ChatOut(ok=False, error="quota"))
                elif r in ("missing_key","empty","retry_exhausted"):
                    job.fut.set_result(ChatOut(ok=False, error=r))
                elif r.startswith("http_"):
                    job.fut.set_result(ChatOut(ok=False, error=r))
                else:
                    job.fut.set_result(ChatOut(ok=False, error="busy"))
        except Exception:
            job.fut.set_result(ChatOut(ok=False, error="exception"))
        finally:
            REQUEST_Q.task_done()

# ---- FastAPI ----
app = FastAPI()

@app.on_event("startup")
async def _startup():
    asyncio.create_task(worker_loop())
    asyncio.create_task(_prune_loop())

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

    key = _key(body)
    if body.reset:
        MEMORY.pop(key, None)

    # build message list: system + history + new user
    history = _hist_get(key)
    messages: List[Dict[str,str]] = [{"role":"system","content":SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role":"user","content":prompt})

    # enqueue and wait bounded
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    try:
        await asyncio.wait_for(REQUEST_Q.put(Job(messages, key, prompt, fut)), timeout=0.5)
    except asyncio.TimeoutError:
        return ChatOut(ok=False, error="busy")

    t0 = time.time()
    try:
        result: ChatOut = await asyncio.wait_for(fut, timeout=REQ_TIMEOUT)
    except asyncio.TimeoutError:
        return ChatOut(ok=False, error="timeout")

    if result.ok and (time.time() - t0) < MIN_DELAY_SECS:
        await asyncio.sleep(MIN_DELAY_SECS - (time.time() - t0))
    return result

@app.post("/v1/reset", response_model=ChatOut)
async def reset(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    MEMORY.pop(_key(body), None)
    return ChatOut(ok=True, reply="memory cleared")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8000")), reload=False)
