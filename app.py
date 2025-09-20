import os, time, asyncio, random, uuid
from typing import Optional, Tuple, Dict, List, Literal
import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

# ===== ENV =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
RPM            = int(os.getenv("RPM", "2"))                  # sustained requests/min
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
OPENAI_TIMEOUT = float(os.getenv("TIMEOUT_SECS", "18"))      # per OpenAI HTTP call
REQ_TIMEOUT    = float(os.getenv("REQ_TIMEOUT_SECS", "45"))  # overall per question
MIN_DELAY_SECS = float(os.getenv("MIN_DELAY_SECS", "0"))
QUEUE_SIZE     = int(os.getenv("QUEUE_SIZE", "128"))
MEM_TURNS      = int(os.getenv("MEMORY_TURNS", "8"))
MEM_TTL_SECS   = int(os.getenv("MEMORY_TTL_SECS", "900"))    # 15 min
JOB_TTL_SECS   = int(os.getenv("JOB_TTL_SECS", "900"))       # 15 min
# =================

SYSTEM_PROMPT = (
    "You are a concise conversational partner for a Roblox NPC. "
    "Mirror the user's language. Keep replies short: 1–2 sentences, 9–22 words. "
    "No links or code unless the user insists repeatedly."
)

# ---- Global pacing ----
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

# ---- Models ----
class AskIn(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    session: Optional[str] = None
    reset: Optional[bool] = False

class AskOut(BaseModel):
    ok: bool
    job_id: Optional[str] = None
    eta: Optional[float] = None
    ready: Optional[bool] = None
    reply: Optional[str] = None
    error: Optional[str] = None

class ResultOut(BaseModel):
    ok: bool
    ready: bool
    reply: Optional[str] = None
    error: Optional[str] = None

# ---- Memory store ----
MEMORY: Dict[str, Dict[str, object]] = {}  # key -> {"msgs":[...], "t":ts}

def _mem_key(user_id: Optional[str], session: Optional[str]) -> str:
    base = (user_id or session or "anon").strip() or "anon"
    return base

def _hist_get(k: str) -> List[Dict[str, str]]:
    rec = MEMORY.get(k)
    if not rec: return []
    return list(rec["msgs"])  # type: ignore

def _hist_reset(k: str):
    MEMORY.pop(k, None)

def _hist_append(k: str, user_text: str, assistant_text: str):
    rec = MEMORY.setdefault(k, {"msgs": [], "t": time.time()})
    msgs: List[Dict[str, str]] = rec["msgs"]  # type: ignore
    msgs.append({"role": "user", "content": user_text})
    msgs.append({"role": "assistant", "content": assistant_text})
    keep = max(0, MEM_TURNS * 2)
    if keep and len(msgs) > keep:
        del msgs[: len(msgs) - keep]
    rec["t"] = time.time()

# ---- Jobs ----
class JobRec(BaseModel):
    id: str
    created: float
    status: Literal["queued", "running", "done", "error"]
    prompt: str
    key: str
    reply: Optional[str] = None
    error: Optional[str] = None

JOBS: Dict[str, JobRec] = {}
REQUEST_Q: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_SIZE)

# ---- OpenAI call with retry ----
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

async def call_openai(messages: List[Dict[str,str]]) -> Tuple[bool, str]:
    if not OPENAI_API_KEY:
        return False, "missing_key"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": MODEL_NAME, "messages": messages, "temperature": 0.6, "max_tokens": 320, "n": 1}

    async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as client:
        attempts = 0
        while attempts < 3:
            attempts += 1
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            if r.status_code == 200:
                data = r.json()
                msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                if not msg: return False, "empty"
                return True, (msg if len(msg) <= 380 else msg[:379] + "…")

            # explicit quota detection
            try:
                data = r.json()
                err_msg = str(data.get("error", {}).get("message", ""))
                if "insufficient_quota" in err_msg or "quota" in err_msg.lower():
                    return False, "quota"
            except:
                pass

            # 429/5xx → back off and retry
            if r.status_code in (429, 500, 502, 503, 504):
                ra = _retry_after_secs(r)
                backoff = ra if ra is not None else min(2 + attempts * 3, 20)
                await asyncio.sleep(backoff + random.random() * 0.3)
                await pace()
                continue

            return False, f"http_{r.status_code}"

        return False, "retry_exhausted"

# ---- Worker ----
async def worker_loop():
    while True:
        job_id = await REQUEST_Q.get()
        job = JOBS.get(job_id)
        if not job:
            REQUEST_Q.task_done()
            continue
        try:
            job.status = "running"  # type: ignore
            await pace()

            history = _hist_get(job.key)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": job.prompt}]

            t0 = time.time()
            ok, r = await call_openai(messages)
            elapsed = time.time() - t0
            if ok:
                job.reply = r
                job.status = "done"  # type: ignore
                _hist_append(job.key, job.prompt, r)
                if elapsed < MIN_DELAY_SECS:
                    await asyncio.sleep(max(0.0, MIN_DELAY_SECS - elapsed))
            else:
                if r == "quota":
                    job.error = "quota"; job.status = "error"  # type: ignore
                elif r in ("missing_key", "empty", "retry_exhausted"):
                    job.error = r; job.status = "error"  # type: ignore
                elif r.startswith("http_"):
                    job.error = r; job.status = "error"  # type: ignore
                else:
                    job.error = "busy"; job.status = "error"  # type: ignore
        except Exception:
            job.error = "exception"; job.status = "error"  # type: ignore
        finally:
            REQUEST_Q.task_done()

# ---- GC ----
async def gc_loop():
    while True:
        now_ts = time.time()
        # jobs
        stale = [jid for jid, rec in JOBS.items() if now_ts - rec.created > JOB_TTL_SECS]
        for jid in stale:
            JOBS.pop(jid, None)
        # memory
        cutoff = now_ts - MEM_TTL_SECS
        mem_stale = [k for k, v in MEMORY.items() if v.get("t", 0) < cutoff]  # type: ignore
        for k in mem_stale:
            MEMORY.pop(k, None)
        await asyncio.sleep(30)

# ---- FastAPI ----
app = FastAPI()

@app.on_event("startup")
async def _startup():
    asyncio.create_task(worker_loop())
    asyncio.create_task(gc_loop())

@app.get("/")
async def root():
    return {"ok": True, "msg": "root alive"}

@app.get("/healthz")
async def healthz():
    return {"ok": True, "queue_depth": REQUEST_Q.qsize(), "jobs": len(JOBS)}

@app.post("/v1/ask", response_model=AskOut)
async def ask(body: AskIn, x_shared_secret: str = Header(default=""), wait_secs: float = Query(default=0.0)):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing_prompt")

    key = _mem_key(body.user_id, body.session)
    if body.reset:
        _hist_reset(key)

    # ETA and early refuse if too deep
    depth = REQUEST_Q.qsize()
    eta = depth * _gap + OPENAI_TIMEOUT + 2
    if eta > REQ_TIMEOUT:
        return AskOut(ok=False, error="busy")

    jid = uuid.uuid4().hex
    JOBS[jid] = JobRec(id=jid, created=time.time(), status="queued", prompt=prompt, key=key)
    try:
        REQUEST_Q.put_nowait(jid)
    except asyncio.QueueFull:
        JOBS.pop(jid, None)
        return AskOut(ok=False, error="busy")

    # Optional synchronous wait-in-ask
    wait_cap = max(0.0, min(wait_secs, REQ_TIMEOUT))
    if wait_cap > 0:
        deadline = time.time() + wait_cap
        while time.time() < deadline:
            rec = JOBS[jid]
            if rec.status == "done":
                return AskOut(ok=True, job_id=jid, ready=True, reply=rec.reply)
            if rec.status == "error":
                return AskOut(ok=False, job_id=jid, ready=True, error=rec.error or "busy")
            await asyncio.sleep(0.3)

    return AskOut(ok=True, job_id=jid, eta=eta, ready=False)

@app.get("/v1/result", response_model=ResultOut)
async def result(job_id: str = Query(...), x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    rec = JOBS.get(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not_found")
    if rec.status == "done":
        return ResultOut(ok=True, ready=True, reply=rec.reply)
    if rec.status == "error":
        return ResultOut(ok=False, ready=True, error=rec.error or "busy")
    # guard against job-level timeout
    if time.time() - rec.created > REQ_TIMEOUT + 10:
        rec.status = "error"; rec.error = "timeout"  # type: ignore
        return ResultOut(ok=False, ready=True, error="timeout")
    return ResultOut(ok=True, ready=False)

@app.post("/v1/reset")
async def reset(body: AskIn, x_shared_secret: str = Header(default="")):
    if not SHARED_SECRET or x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    _hist_reset(_mem_key(body.user_id, body.session))
    return {"ok": True, "reply": "memory cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8000")), reload=False)
