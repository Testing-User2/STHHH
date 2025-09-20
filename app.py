# app.py — deadline-aware queue worker (no false timeouts)
import os, time, asyncio, random
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ========= ENV =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
RPM            = int(os.getenv("RPM", "2"))                    # sustained requests/min
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
OPENAI_TIMEOUT = float(os.getenv("TIMEOUT_SECS", "18"))        # per OpenAI HTTP request
REQ_TIMEOUT    = float(os.getenv("REQ_TIMEOUT_SECS", "150"))   # end-to-end cap per /v1/chat
QUEUE_SIZE     = int(os.getenv("QUEUE_SIZE", "256"))           # bounded queue
HOST           = os.getenv("HOST", "0.0.0.0")
PORT           = int(os.getenv("PORT", "8000"))
MIN_DELAY_SECS = float(os.getenv("MIN_DELAY_SECS", "0"))
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

async def pace_with_deadline(deadline: float) -> bool:
    """Sleep to enforce RPM only if there is enough budget left.
    Returns False if we cannot respect RPM within the remaining time.
    """
    global _last_call
    async with _gate_lock:
        now = time.time()
        wait = _gap - (now - _last_call)
        if wait <= 0:
            _last_call = now
            return True
        remaining = deadline - now
        if remaining <= wait + 0.5:
            # Not enough time left to wait for RPM gap + minimal slack
            return False
        await asyncio.sleep(wait)
        _last_call = time.time()
        return True

# ---- models ----
class ChatIn(BaseModel):
    prompt: str

class ChatOut(BaseModel):
    ok: bool
    reply: Optional[str] = None
    error: Optional[str] = None

# ---- utilities ----
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
            import re
            m = re.search(r"(\d+)\s*s", msg)
            if m: return float(m.group(1))
    except: pass
    return None

# ---- OpenAI call with deadline ----
async def call_openai(prompt: str, deadline: float) -> Tuple[bool, str]:
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
    attempts = 0
    async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as client:
        while True:
            attempts += 1
            # Ensure we have enough time left for one HTTP call
            remaining = deadline - time.time()
            if remaining <= 1.0:
                return False, "timeout"

            try:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            except httpx.TimeoutException:
                # Try again if we still have time
                if deadline - time.time() > 5 and attempts < 3:
                    continue
                return False, "timeout"

            if r.status_code == 200:
                data = r.json()
                msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                if not msg: return False, "empty"
                if len(msg) > 380: msg = msg[:379] + "…"
                return True, msg

            # explicit quota
            try:
                data = r.json(); err_msg = str(data.get("error", {}).get("message", ""))
                if "insufficient_quota" in err_msg or "quota" in err_msg.lower():
                    return False, "quota"
            except: pass

            if r.status_code in (429, 500, 502, 503, 504):
                ra = _retry_after_secs(r)
                # choose backoff but do not exceed deadline
                base = ra if ra is not None else min(2 + attempts * 3, 20)
                backoff = base + random.random() * 0.3
                remaining = deadline - time.time()
                if remaining <= backoff + 1.0 or attempts >= 3:
                    # not enough time left or too many attempts
                    return False, "timeout"
                await asyncio.sleep(backoff)
                continue

            return False, f"http_{r.status_code}"

# ---- queue + worker ----
class Job:
    __slots__ = ("prompt","fut","deadline")
    def __init__(self, prompt: str, fut: asyncio.Future, deadline: float):
        self.prompt = prompt
        self.fut = fut
        self.deadline = deadline

REQUEST_Q: asyncio.Queue[Job] = asyncio.Queue(maxsize=QUEUE_SIZE)

async def worker_loop():
    while True:
        job = await REQUEST_Q.get()
        try:
            now = time.time()
            remaining = job.deadline - now
            if remaining <= 1.0:
                job.fut.set_result(ChatOut(ok=False, error="timeout"))
                continue

            # Respect RPM only if feasible within deadline; otherwise return "busy"
            can_wait = await pace_with_deadline(job.deadline)
            if not can_wait:
                job.fut.set_result(ChatOut(ok=False, error="busy"))
                continue

            ok, r = await call_openai(job.prompt, job.deadline)
            if ok:
                # Optional min-delay shaping for consistent feel
                elapsed = time.time() - (job.deadline - REQ_TIMEOUT)
                if MIN_DELAY_SECS > 0 and elapsed < MIN_DELAY_SECS:
                    await asyncio.sleep(MIN_DELAY_SECS - elapsed)
                job.fut.set_result(ChatOut(ok=True, reply=r))
            else:
                if r in ("quota","missing_key","empty"):
                    job.fut.set_result(ChatOut(ok=False, error=r))
                elif r.startswith("http_"):
                    job.fut.set_result(ChatOut(ok=False, error=r))
                else:
                    job.fut.set_result(ChatOut(ok=False, error="timeout"))
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
    return {"ok": True, "queue_depth": REQUEST_Q.qsize(), "gap_s": _gap}

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

    # admission: only reject if bounded queue is full
    if REQUEST_Q.full():
        return ChatOut(ok=False, error="busy")

    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    deadline = time.time() + REQ_TIMEOUT
    await REQUEST_Q.put(Job(prompt, fut, deadline))

    try:
        result: ChatOut = await asyncio.wait_for(fut, timeout=REQ_TIMEOUT + 1.0)
    except asyncio.TimeoutError:
        return ChatOut(ok=False, error="timeout")

    if not result.ok:
        # compact log without spamming
        print(f"[chat] non-ok -> {result.error} (depth_now={REQUEST_Q.qsize()})", flush=True)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
