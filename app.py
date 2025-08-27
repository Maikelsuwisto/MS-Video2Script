from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import os, shutil, uuid

app = FastAPI(title="MS-Video2Script Backend")

# -------------------------
# CORS (optional for dev)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Serve React build
# -------------------------
build_path = "build"  # build folder is in the same directory as app.py
assets_path = os.path.join(build_path, "assets")

# Ensure build folder exists
if not os.path.exists(build_path):
    raise RuntimeError(f"React build folder not found at '{build_path}'")

# Serve React frontend
app.mount("/", StaticFiles(directory=build_path, html=True), name="frontend")
app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

@app.get("/{full_path:path}")
async def serve_react(full_path: str, request: Request):
    file_path = os.path.join(build_path, full_path)
    if not os.path.exists(file_path):
        file_path = os.path.join(build_path, "index.html")
    response = FileResponse(file_path)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

# -------------------------
# Uploads folder
# -------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# Whisper model
# -------------------------
model = WhisperModel("tiny", device="cpu", compute_type="int8")

# -------------------------
# Transcription API
# -------------------------
@app.post("/transcribe")
async def transcribe(video: UploadFile = File(...), with_timestamps: str = Form("0")):
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        segments, _ = model.transcribe(save_path, beam_size=5)
        transcription = []
        for seg in segments:
            entry = {"text": seg.text}
            if with_timestamps == "1":
                entry["start"] = f"{seg.start:.2f}"
                entry["end"] = f"{seg.end:.2f}"
            transcription.append(entry)
        return {"transcription": transcription, "audio_url": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
