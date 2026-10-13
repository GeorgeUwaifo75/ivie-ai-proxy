# 🗣️ Ivie AI — Talking Avatar (FastAPI + Render)

A port of the original `avatar.html` talking-avatar app to a **FastAPI**
backend with the same zero-dependency frontend, deployable on
[Render.com](https://render.com).

## Features

- 🎭 Animated SVG avatar with real-time, phoneme-aware lip-sync
- 🎙️ West-African female voice prioritisation (Nigerian → West Africa → Africa → English)
- 💬 `/chat` API — OpenAI-compatible or offline `echo` mode
- 🧠 Session memory (`MAX_HISTORY_TURNS`)
- 📥 Chat export (JSON / TXT)
- ✨ New Session button
- ⚡ Health check at `/health` for Render

## Quick start (local)

```bash
git clone <your-repo>
cd ivie-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # optional — defaults to offline "echo" mode
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>.

## Deploy to Render

### Option A — Blueprint (`render.yaml`)

1. Push this repo to GitHub.
2. On Render → **New** → **Blueprint** → pick your repo.
3. Render auto-creates the service from `render.yaml`.
4. (Optional) In **Environment**, add `OPENAI_API_KEY` and set
   `LLM_PROVIDER=openai`.
5. Done. First boot takes ~1 min.

### Option B — Manual web service

| Field | Value |
|---|---|
| Environment | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

## API

### `GET /health`
```json
{ "status": "ok", "service": "Ivie AI", "provider": "echo" }
```

### `POST /chat`
```json
// request
{ "message": "Hello Ivie!", "session_id": "user-42" }

// response
{ "reply": "Hello! I'm Ivie...", "session_id": "user-42", "model": "echo" }
```

### `POST /reset`
Clears the session's conversation history.

## Environment variables

| Var | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `echo` | `echo` or `openai` |
| `OPENAI_API_KEY` | — | Required when provider = `openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | |
| `OPENAI_BASE_URL` | — | For Groq/Together/Azure/Ollama |
| `ALLOWED_ORIGINS` | `*` | Comma-separated |
| `MAX_HISTORY_TURNS` | `8` | |
| `SYSTEM_PROMPT` | (Nigerian Ivie persona) | |

## License

MIT