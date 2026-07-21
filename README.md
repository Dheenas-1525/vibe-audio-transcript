# ViBe Audio Transcript

**v1.3.1** — YouTube URL → timestamped transcript → template-driven question bank CSV.

Paste a YouTube URL, get a timestamped transcript (download as TXT / SRT / VTT). Upload a
question-bank template CSV and generate matching questions from that transcript using a
self-hosted LLM (Qwen3, via vLLM's OpenAI-compatible API) — download the result as a CSV
that matches your template's exact columns.

Transcription runs fully local (`yt-dlp` + `faster-whisper`, no external API). Question
generation calls out to your own vLLM server — nothing else leaves the machine.

📖 **Full manual:** open [`MANUAL.html`](./MANUAL.html) in a browser for architecture,
setup, configuration, the API reference, troubleshooting, and version history.

## Run

```sh
cp .env.example .env   # fill in VLLM_API_BASE and VLLM_MODEL
docker compose up --build -d
```

Open `http://localhost:8000` (or forward that port via VS Code Remote-SSH / an SSH tunnel
if running on a remote server — see `MANUAL.html` § Accessing the app).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny` \| `base` \| `small` \| `medium` \| `large-v3` — bigger = slower, more accurate |
| `WHISPER_DEVICE` | `cpu` | `cpu` \| `cuda` — on a GPU host also run with `docker-compose.gpu.yml` (see `MANUAL.html`) |
| `WHISPER_COMPUTE_TYPE` | `int8` | CTranslate2 quantization — `int8` for CPU, `float16` recommended for GPU |
| `VLLM_API_BASE` | *(required)* | Base URL of your vLLM server, e.g. `http://172.16.13.91:8002/v1` — no `/chat/completions` suffix |
| `VLLM_API_KEY` | `EMPTY` | vLLM API key, if your server checks one |
| `VLLM_MODEL` | *(required)* | Exact served model ID, e.g. `Qwen/Qwen3-30B-A3B` |
| `WEB_PORT` | `8000` | Host-side port (change if 8000 is already taken) |
| `JOBS_DB_PATH` | `/app/data/jobs.db` | SQLite file persisting job state across restarts (backed by the `app-data` volume) |

The Whisper model downloads on first use and is cached in a Docker volume.

## Deploying an update

```sh
# locally
git add -A && git commit -m "..." && git push

# on the server
git pull && docker compose up --build -d
```

## Security

This app has **no authentication**. Anyone who can reach its port can trigger YouTube
downloads and LLM calls on your server. Keep it behind an SSH tunnel / private network
unless you add an auth layer first.

---
See [`MANUAL.html`](./MANUAL.html) for the full end-to-end guide, including the question
bank CSV format, API reference, and version history.
