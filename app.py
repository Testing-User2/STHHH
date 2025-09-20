import os, time, asyncio, random, re
from typing import Optional, Tuple
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ===== ENV (Render) =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
SHARED_SECRET  = os.getenv("SHARED_SECRET", "")
# Hard wall per request; keep generous so OpenAI can sometimes win, we still reply at ~2s.
REQ_TIMEOUT_SECS          = float(os.getenv("REQ_TIMEOUT_SECS", "60"))
# How long we give OpenAI before we fall back (must be < FAST_REPLY_SECS - 0.2)
LLM_SOFT_DEADLINE_SECS    = float(os.getenv("LLM_SOFT_DEADLINE_SECS", "2.5"))
# Target user-visible latency (seconds) — your 2s requirement
FAST_REPLY_SECS           = float(os.getenv("FAST_REPLY_SECS", "3.0"))
# Per-attempt HTTP timeout when calling OpenAI (cap per socket read)
OPENAI_ATTEMPT_TIMEOUT_S  = float(os.getenv("OPENAI_ATTEMPT_TIMEOUT_SECS", "2.5"))
HOST                      = os.getenv("HOST", "0.0.0.0")
PORT                      = int(os.getenv("PORT", "8000"))
# ========================

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
            m = re.search(r"(\d+)\s*s", msg)
            if m: return float(m.group(1))
    except: pass
    return None

async def call_openai_soft(prompt: str) -> Tuple[bool, str, str]:
    """Try once with a short timeout. On success -> (True, text, ''). On fail -> (False, '', reason)."""
    if not OPENAI_API_KEY:
        return False, "", "missing_key"
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
    timeout = httpx.Timeout(
        connect=min(5.0, OPENAI_ATTEMPT_TIMEOUT_S),
        read=OPENAI_ATTEMPT_TIMEOUT_S,
        write=5.0,
        pool=5.0,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
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
    # explicit quota
    try:
        data = r.json(); err_msg = str(data.get("error", {}).get("message", ""))
        if "insufficient_quota" in err_msg.lower():
            return False, "", "quota"
    except Exception:
        pass
    if r.status_code in (429, 500, 502, 503, 504):
        return False, "", f"transient_{r.status_code}"
    return False, "", f"http_{r.status_code}"

# ---- Fallback synthesis (short, neutral, no apologies) ----
_DE_WORDS = {"und","oder","nicht","ich","du","wir","ihr","der","die","das","ein","kein","bitte","danke","warum","wie","was"}
def _is_german(s: str) -> bool:
    s_low = s.lower()
    if any(ch in s_low for ch in ("ä","ö","ü","ß")): return True
    tokens = re.findall(r"[a-zA-ZäöüÄÖÜß]+", s_low)
    hits = sum(1 for t in tokens if t in _DE_WORDS)
    return hits >= 2

def _clip_words(s: str, n: int) -> str:
    words = re.findall(r"[^\s]+", s)
    return " ".join(words[:n])

def fallback_reply(prompt: str) -> str:
    german = _is_german(prompt)
    w = re.sub(r"\s+", " ", prompt.strip())
    core = _clip_words(w, 9)
    if german:
        forms = [
            f"Kurzer Take: {core}. Nächster Schritt in zwei Worten, dann los.",
            f"Fix: {core}. Wenn’s nicht passt, sag’s knapper.",
            f"Klar: {core}. Willst du’s präziser, sag genauer.",
            f"Okay: {core}. Zwei Sätze reichen, weiter?",
        ]
    else:
        forms = [
            f"Quick take: {core}. One concrete step, then go.",
            f"Fast: {core}. If off, tighten it.",
            f"Okay: {core}. Two lines max, proceed.",
            f"Note: {core}. Short next step, then move.",
        ]
    out = random.choice(forms)
    # ensure 9–22 words, 1–2 sentences
    if len(out.split()) < 9:
        out += " Weiter kurz." if german else " Keep it tight."
    return out[:380]

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

    t0 = time.time()
    deadline = t0 + REQ_TIMEOUT_SECS
    llm_deadline = min(deadline - 0.5, t0 + max(0.5, min(LLM_SOFT_DEADLINE_SECS, FAST_REPLY_SECS - 0.4)))

    # Try model quickly
    ok, text, reason = await call_openai_soft(prompt)
    if not ok:
        # Log once per failure path (compact)
        print(f"[chat] LLM miss -> {reason}", flush=True)
        text = fallback_reply(prompt)

    # Hold until FAST_REPLY_SECS for consistent feel
    elapsed = time.time() - t0
    wait = max(0.0, FAST_REPLY_SECS - elapsed)
    if wait > 0:
        await asyncio.sleep(wait)

    return ChatOut(ok=True, reply=text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
