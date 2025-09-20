import os
import asyncio
import time
import httpx
import discord
from discord.ext import commands

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
RPM            = int(os.getenv("RPM", "3"))
TIMEOUT_SECS   = float(os.getenv("TIMEOUT_SECS", "12"))
DISCORD_TOKEN  = os.getenv("DISCORD_TOKEN", "")

SYSTEM_PROMPT = (
    "You are a concise conversational partner. "
    "Mirror the user's language. Keep replies short: 1–2 sentences."
)

_gap = 60.0 / max(1, RPM)
_last = 0.0
_lock = asyncio.Lock()

async def pace():
    global _last
    async with _lock:
        now = time.time()
        wait = _gap - (now - _last)
        if wait > 0:
            await asyncio.sleep(wait)
        _last = time.time()

async def ask_openai(prompt: str) -> str:
    await pace()
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 200,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT_SECS) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        if r.status_code == 429:
            await pace()
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    msg = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return msg or "No content."

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!ai ", intents=intents, help_command=None)

@bot.event
async def on_ready():
    print(f"Discord bot ready: {bot.user}")

@bot.command(name="", pass_context=True)
async def ask(ctx: commands.Context, *, q: str = ""):
    if not q:
        await ctx.reply("question missing"); return
    try:
        reply = await ask_openai(q)
        await ctx.reply(reply[:1900])
    except Exception as e:
        await ctx.reply(f"error: {type(e).__name__}")

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_TOKEN missing")
    bot.run(DISCORD_TOKEN)
