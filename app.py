import os
import uuid
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# -----------------------
# Debug print
# -----------------------
print("Starting FastAPI backend...")

app = FastAPI(title="MS-Video2Script Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Serve React frontend
# -----------------------
build_path = "build"  # React build folder in same folder as app.py
assets_path = os.path.join(build_path, "assets")

if not os.path.exists(build_path):
    raise RuntimeError(f"React build folder not found at '{build_path}'")

app.mount("/", StaticFiles(directory=build_path, html=True), name="frontend")
app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# -----------------------
# Upload folder
# -----------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------
# Whisper model (lazy load)
# -----------------------
model = None  # will load on first transcription

def get_model():
    global model
    if model is None:
        print("Loading Whisper tiny model...")  # debug
        from faster_whisper import WhisperModel
        try:
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("Model loaded successfully.")  # debug
        except Exception as e:
            print("Error loading Whisper model:", e)
            raise e
    return model

# -----------------------
# Helper: Convert seconds to HH:MM:SS
# -----------------------
def seconds_to_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# -----------------------
# Transcription endpoint
# -----------------------
@app.post("/transcribe")
async def transcribe(
    video: UploadFile = File(...),
    with_timestamps: str = Form("0")
):
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Save uploaded video temporarily
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Load model lazily
        whisper_model = get_model()

        # Run transcription
        segments, info = whisper_model.transcribe(save_path, beam_size=5)

        transcription = []
        for seg in segments:
            entry = {"text": seg.text}
            if with_timestamps == "1":
                entry["start"] = seconds_to_hms(seg.start)
                entry["end"] = seconds_to_hms(seg.end)
            transcription.append(entry)

        # Optional: generate audio URL
        audio_url = None

        return {"transcription": transcription, "audio_url": audio_url}

    except Exception as e:
        print("Error during transcription:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process transcription: {str(e)}")

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

# -----------------------
# Debug Exception Handler
# -----------------------
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc()
        },
    )

# -----------------------
# Health check
# -----------------------
@app.get("/health")
def health():
    return {"message": "MS-Video2Script API is running ✅"}
