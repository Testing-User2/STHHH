# Roblox ↔ LLM7 bridge
SHARED_SECRET=your-shared-secret

# LLM7 (OpenAI-compatible)
LLM7_BASE=https://llm7.io/v1
LLM7_API_KEY=unused         # or your token from https://token.llm7.io

# Model + decoding
MODEL_NAME=gpt-4
TEMP=0.6
MAX_TOKENS=160

# Timeouts / retries
ATTEMPT_TIMEOUT_SECS=8.0
REQ_TIMEOUT_SECS=25
MAX_RETRIES=2
RETRY_BACKOFF_SECS=0.8

# Server bind
HOST=0.0.0.0
# PORT is provided by Render
