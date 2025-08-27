import os
import uuid
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi.responses import JSONResponse, FileResponse  # <-- add FileResponse here


print("Starting FastAPI backend...")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="MS-Video2Script Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_coop_coep_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

# -----------------------
# Serve React SPA at root
# -----------------------
build_path = "build"  # React build folder in same folder as app.py
if not os.path.exists(build_path):
    raise RuntimeError(f"React build folder not found at '{build_path}'")

app.mount("/assets", StaticFiles(directory=os.path.join(build_path, "assets")), name="assets")

# Catch-all route for SPA
@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    return FileResponse(os.path.join(build_path, "index.html"))

# -----------------------
# Health checks
# -----------------------
@app.get("/health")
def health():
    return {"message": "MS-Video2Script API is running ✅"}

@app.get("/")
def root():
    return {"message": "Backend running ✅"}

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
        print("Loading Whisper tiny model...")
        from faster_whisper import WhisperModel
        try:
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading Whisper model:", e)
            raise e
    return model

# -----------------------
# Helper: seconds -> HH:MM:SS
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

        return {"transcription": transcription}

    except Exception as e:
        print("Error during transcription:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process transcription: {str(e)}")

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
            

# -----------------------
# Debug exception handler
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

if __name__ == "__main__":
    port = int(os.environ["PORT"])
    print(f"Starting Uvicorn on Railway port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port)

