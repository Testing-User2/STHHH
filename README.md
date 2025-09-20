<!-- server/README.md -->
# ChatNPC AI Server

Single-file FastAPI that returns a reply in ~2.6 s using Groq (default) or OpenAI.

## Deploy (Render)
1. Create a Web Service from this folder.
2. Start command:  
   `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Set **Environment Variables** in Render:
   - Choose **one** provider:
     - Groq:
       - `PROVIDER=groq`
       - `GROQ_API_KEY=...`
       - `MODEL_NAME=llama3-8b-8192`
     - OpenAI:
       - `PROVIDER=openai`
       - `OPENAI_API_KEY=...`
       - `MODEL_NAME=gpt-4o-mini`
   - Common:
     - `SHARED_SECRET=<same as Roblox>`
     - `FAST_REPLY_SECS=2.6`
     - `ATTEMPT_TIMEOUT_SECS=2.2`
     - `REQ_TIMEOUT_SECS=45`
     - `HOST=0.0.0.0`
4. After deploy, test:
