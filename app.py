# app.py
import os, time, asyncio, random
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ===== ENV =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
RPM            = int(os.getenv("RPM", "3"))                  # sustained requests/min
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
OPENAI_TIMEOUT = float(os.getenv("TIMEOUT_SECS", "18"))      # per OpenAI request
REQ_TIMEOUT    = float(os.getenv("REQ_TIMEOUT_SECS", "120")) # end-to-end cap
MIN_DELAY_SECS = float(os.getenv("MIN_DELAY_SECS", "0"))
QUEUE_SIZE     = int(os.getenv("QUEUE_SIZE", "128"))
HOST           = os.getenv("HOST", "0.0.0.0")
PORT           = int(os.getenv("PORT", "8000"))
# =================

SYSTEM_PROMPT = (
    "You are a concise conversational partner for a Roblox NPC. "
    "Mirror the user's language. Keep replies short: 1–2 sentences, 9–22 words. "
    "No links or code unless the user insists repeatedly."
)

# ---- pacing (uniform gap) ----
_gap = 60.0 / max(1, RPM)
_last_call = 0.0
_gate_lock = asyncio.Lock()

def current_gate_wait() -> float:
    return max(0.0, _gap - (time.time() - _last_call))

async def pace():
    global _last_call
    async with _gate_lock:
        rem = current_gate_wait()
        if rem > 0:
            await asyncio.sleep(rem)
        _last_call = time.time()

def expected_eta(depth: int) -> float:
    # Wait for N jobs ahead + current gate + upstream call + small overhead
    return depth * _gap + current_gate_wait() + OPENAI_TIMEOUT + 2.0

# ---- models ----
class ChatIn(BaseModel):
    prompt: str

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

# ---- OpenAI call ----
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
            m = str(j.get("error", {}).get("message", ""))
            import re
            z = re.search(r"(\d+)\s*s", m)
            if z: return float(z.group(1))
    except: pass
    return None

async def call_openai(prompt: str) -> Tuple[bool, str]:
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
    async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as client:
        attempts = 0
        while attempts < 3:
            attempts += 1
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            if r.status_code == 200:
                data = r.json()
                msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                if not msg: return False, "empty"
                if len(msg) > 380: msg = msg[:379] + "…"
                return True, msg
            try:
                data = r.json(); err_msg = str(data.get("error", {}).get("message", ""))
                if "insufficient_quota" in err_msg or "quota" in err_msg.lower():
                    return False, "quota"
            except: pass
            if r.status_code in (429, 500, 502, 503, 504):
                ra = _retry_after_secs(r)
                backoff = ra if ra is not None else min(2 + attempts * 3, 20)
                await asyncio.sleep(backoff + random.random() * 0.3)
                await pace()
                continue
            return False, f"http_{r.status_code}"
        return False, "retry_exhausted"

# ---- queue + worker ----
class Job:
    __slots__ = ("prompt","fut")
    def __init__(self, prompt: str, fut: asyncio.Future):
        self.prompt = prompt; self.fut = fut

REQUEST_Q: asyncio.Queue[Job] = asyncio.Queue(maxsize=QUEUE_SIZE)

async def worker_loop():
    while True:
        job = await REQUEST_Q.get()
        try:
            await pace()
            ok, r = await call_openai(job.prompt)
            if ok:
                job.fut.set_result(ChatOut(ok=True, reply=r))
            else:
                if r == "quota": job.fut.set_result(ChatOut(ok=False, error="quota"))
                elif r in ("missing_key","empty","retry_exhausted"): job.fut.set_result(ChatOut(ok=False, error=r))
                elif r.startswith("http_"): job.fut.set_result(ChatOut(ok=False, error=r))
                else: job.fut.set_result(ChatOut(ok=False, error="busy"))
        except Exception:
            job.fut.set_result(ChatOut(ok=False, error="exception"))
        finally:
            REQUEST_Q.task_done()

app = FastAPI()

@app.on_event("startup")
async def _startup():
    asyncio.create_task(worker_loop())

@app.api_route("/", methods=["GET","HEAD"])
async def root():
    return {"ok": True, "msg": "root alive", "queue_depth": REQUEST_Q.qsize()}

@app.api_route("/healthz", methods=["GET","HEAD"])
async def healthz():
    return {"ok": True, "queue_depth": REQUEST_Q.qsize()}

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    depth = REQUEST_Q.qsize()
    eta = expected_eta(depth)
    if eta > REQ_TIMEOUT:
        result = ChatOut(ok=False, error="busy")
        print(f"[chat] non-ok -> {result.error} (eta={eta:.1f}s, depth={depth}, gap={_gap:.1f}s)", flush=True)
        return result

    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    try:
        await asyncio.wait_for(REQUEST_Q.put(Job(prompt, fut)), timeout=0.5)
    except asyncio.TimeoutError:
        result = ChatOut(ok=False, error="busy")
        print(f"[chat] non-ok -> {result.error} (enqueue timeout)", flush=True)
        return result

    try:
        result: ChatOut = await asyncio.wait_for(fut, timeout=eta + 2.0)
    except asyncio.TimeoutError:
        result = ChatOut(ok=False, error="timeout")
        print(f"[chat] non-ok -> {result.error} (eta={eta:.1f}s exceeded)", flush=True)
        return result

    if not result.ok:
        print(f"[chat] non-ok -> {result.error}", flush=True)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
