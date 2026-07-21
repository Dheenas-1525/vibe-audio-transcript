FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JS runtime to decrypt YouTube's current signature scheme —
# without it, downloads increasingly fail with 403 Forbidden.
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade -r requirements.txt

COPY app.py .
COPY qb_generator.py .
COPY static ./static

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
