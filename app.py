import csv
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import uuid

import yt_dlp
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from openai import OpenAI
from pydantic import BaseModel

import qb_generator

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
VLLM_API_BASE = os.environ.get("VLLM_API_BASE")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.environ.get("VLLM_MODEL")
DB_PATH = os.environ.get("JOBS_DB_PATH", "/app/data/jobs.db")

app = FastAPI(title="ViBe Audio Transcript")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)
jobs: dict[str, dict] = {}
model_lock = threading.Lock()
_model = None
vllm_client_lock = threading.Lock()
_vllm_client = None
db_lock = threading.Lock()

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db_conn.execute("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, data TEXT NOT NULL)")
db_conn.commit()


def save_job(job_id: str):
    with db_lock:
        db_conn.execute(
            "INSERT INTO jobs (job_id, data) VALUES (?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET data = excluded.data",
            (job_id, json.dumps(jobs[job_id])),
        )
        db_conn.commit()


def load_jobs():
    """Restore jobs from disk on startup; any job whose background thread
    died with the previous process can never finish, so mark it errored
    instead of leaving it stuck showing "transcribing"/"generating" forever."""
    with db_lock:
        rows = db_conn.execute("SELECT job_id, data FROM jobs").fetchall()
    for job_id, data in rows:
        job = json.loads(data)
        if job.get("status") not in ("done", "error"):
            job["status"] = "error"
            job["error"] = "Interrupted by a server restart — please re-transcribe."
        if job.get("qb_status") not in (None, "done", "error"):
            job["qb_status"] = "error"
            job["qb_error"] = "Interrupted by a server restart — please regenerate."
        jobs[job_id] = job


load_jobs()


def get_model():
    global _model
    with model_lock:
        if _model is None:
            _model = WhisperModel(
                MODEL_SIZE, device="cpu", compute_type="int8",
                cpu_threads=os.cpu_count() or 4,
            )
        return _model


def get_vllm_client():
    global _vllm_client
    with vllm_client_lock:
        if _vllm_client is None:
            if not VLLM_API_BASE or not VLLM_MODEL:
                raise RuntimeError("VLLM_API_BASE and VLLM_MODEL must be set")
            _vllm_client = OpenAI(base_url=VLLM_API_BASE, api_key=VLLM_API_KEY)
        return _vllm_client


def fmt_ts(seconds: float, vtt: bool = False) -> str:
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = seconds % 60
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int(s % 1 * 1000):03d}"


def run_job(job_id: str, url: str):
    job = jobs[job_id]
    tmpdir = tempfile.mkdtemp()
    try:
        job["status"] = "downloading"
        save_job(job_id)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        job["title"] = info.get("title", "transcript")
        audio = os.path.join(tmpdir, "audio.mp3")

        job["status"] = "transcribing"
        save_job(job_id)
        segments, _ = get_model().transcribe(audio, vad_filter=True)
        result = []
        for seg in segments:
            result.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            job["progress"] = round(seg.end, 1)
        job["segments"] = result
        job["status"] = "done"
        save_job(job_id)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        save_job(job_id)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class TranscribeRequest(BaseModel):
    url: str


@app.post("/api/transcribe")
def transcribe(req: TranscribeRequest):
    if not re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", req.url):
        raise HTTPException(400, "Not a valid YouTube URL")
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "progress": 0}
    save_job(job_id)
    threading.Thread(target=run_job, args=(job_id, req.url), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/download/{job_id}")
def download(job_id: str, fmt: str = "txt"):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Transcript not ready")
    segs = job["segments"]
    if fmt == "srt":
        body = "\n".join(
            f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text']}\n"
            for i, s in enumerate(segs, 1)
        )
        media = "application/x-subrip"
    elif fmt == "vtt":
        body = "WEBVTT\n\n" + "\n".join(
            f"{fmt_ts(s['start'], vtt=True)} --> {fmt_ts(s['end'], vtt=True)}\n{s['text']}\n"
            for s in segs
        )
        media = "text/vtt"
    else:
        fmt = "txt"
        body = "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in segs)
        media = "text/plain"
    name = re.sub(r"[^\w\- ]", "", job.get("title", "transcript"))[:60] or "transcript"
    return PlainTextResponse(
        body, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"'},
    )


def run_qb_job(job_id: str, template_columns: list[str], questions_per_segment: int):
    job = jobs[job_id]
    try:
        job["qb_status"] = "generating"
        save_job(job_id)
        client = get_vllm_client()

        def progress(done, total):
            job["qb_progress"] = {"done": done, "total": total}
            save_job(job_id)

        rows = qb_generator.generate_question_bank(
            job["segments"], template_columns, client, VLLM_MODEL,
            questions_per_segment=questions_per_segment, progress_cb=progress,
        )
        job["qb_rows"] = rows
        job["qb_columns"] = template_columns
        job["qb_status"] = "done"
        save_job(job_id)
    except Exception as e:
        job["qb_status"] = "error"
        job["qb_error"] = str(e)
        save_job(job_id)


@app.post("/api/generate-questions/{job_id}")
async def generate_questions(
    job_id: str,
    template: UploadFile = File(...),
    questions_per_segment: int = Form(5),
):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Transcript not ready")
    content = (await template.read()).decode("utf-8-sig")
    try:
        template_columns = next(csv.reader(io.StringIO(content)))
    except StopIteration:
        raise HTTPException(400, "Template CSV is empty")
    job["qb_status"] = "queued"
    job["qb_progress"] = {"done": 0, "total": 0}
    save_job(job_id)
    threading.Thread(
        target=run_qb_job, args=(job_id, template_columns, questions_per_segment), daemon=True
    ).start()
    return {"job_id": job_id}


@app.get("/api/qb-status/{job_id}")
def qb_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "qb_status": job.get("qb_status"),
        "qb_progress": job.get("qb_progress"),
        "qb_error": job.get("qb_error"),
    }


@app.get("/api/download-questions/{job_id}")
def download_questions(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("qb_status") != "done":
        raise HTTPException(404, "Question bank not ready")
    body = qb_generator.rows_to_csv_str(job["qb_rows"], job["qb_columns"])
    name = re.sub(r"[^\w\- ]", "", job.get("title", "questions"))[:60] or "questions"
    return PlainTextResponse(
        body, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}_QB.csv"'},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
