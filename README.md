# ViBe Audio Transcript

Paste a YouTube URL, get a timestamped transcript, download as TXT / SRT / VTT.
Runs fully local: yt-dlp for audio, faster-whisper for speech-to-text.

## Run

```sh
docker compose up --build
```

Open http://localhost:8000

## Accuracy vs speed

Set the model in `docker-compose.yml` (or `WHISPER_MODEL=... docker compose up`):

| Model    | Speed   | Accuracy |
|----------|---------|----------|
| tiny     | fastest | lowest   |
| small    | good    | good (default) |
| medium   | slower  | better   |
| large-v3 | slowest | best     |

The model downloads on first use and is cached in a Docker volume.
