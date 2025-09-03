import os
import re
import uuid
import shutil
import traceback
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydub import AudioSegment
import webrtcvad

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
build_path = "build"
if not os.path.exists(build_path):
    raise RuntimeError(f"React build folder not found at '{build_path}'")

app.mount("/assets", StaticFiles(directory=os.path.join(build_path, "assets")), name="assets")

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
model = None

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
# WebRTC VAD functions
# -----------------------
ffmpeg_path = "ffmpeg"  # ensure ffmpeg is in PATH or provide full path

def extract_audio_ffmpeg(video_path, audio_path="audio.wav", progress_callback=None):
    if progress_callback:
        progress_callback(0.1, "🔊 Extracting audio...")
    cmd = [ffmpeg_path, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {process.stderr.decode()}")
    if progress_callback:
        progress_callback(0.25, "✅ Audio extracted.")
    return audio_path

def detect_first_speech_offset_webrtcvad(audio_path, aggressiveness=3, min_speech_ms=1000, ignore_before_ms=1000):
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    raw_audio = audio.raw_data
    vad = webrtcvad.Vad(aggressiveness)
    frame_duration = 30  # ms
    frame_size = int(16000 * frame_duration / 1000) * 2
    frames = [raw_audio[i:i+frame_size] for i in range(0, len(raw_audio), frame_size)]

    speech_start_frame = None
    speech_frames_count = 0
    required_speech_frames = min_speech_ms // frame_duration
    ignore_before_frames = ignore_before_ms // frame_duration

    for i, frame in enumerate(frames):
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate=16000):
            if speech_start_frame is None:
                speech_start_frame = i
            speech_frames_count += 1
        else:
            if speech_start_frame is not None and speech_frames_count >= required_speech_frames:
                if speech_start_frame >= ignore_before_frames:
                    return (speech_start_frame * frame_duration) / 1000.0
            speech_start_frame = None
            speech_frames_count = 0

    if speech_start_frame is not None and speech_frames_count >= required_speech_frames:
        if speech_start_frame >= ignore_before_frames:
            return (speech_start_frame * frame_duration) / 1000.0

    return 0.0

def detect_first_speech(video_path):
    try:
        audio_path = extract_audio_ffmpeg(video_path)
        offset_sec = detect_first_speech_offset_webrtcvad(audio_path)
        return offset_sec
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# -----------------------
# Helper: smooth splitting
# -----------------------
def split_segment_text_precise(text, start_time, end_time):
    """
    Split a Whisper segment text into sentences with proportional timestamps.
    """
    # Regex to split while keeping delimiters
    sentences = re.split(r'([.!?])', text)

    # Recombine delimiters with sentence parts
    sentences = [
        (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else ""))
        for i in range(0, len(sentences), 2)
    ]

    # Clean up and remove empties
    sentences = [s.strip() for s in sentences if s.strip()]

    # Allocate time proportionally
    total_chars = sum(len(s) for s in sentences)
    results = []
    current_time = start_time

    for s in sentences:
        proportion = len(s) / total_chars if total_chars > 0 else 0
        duration = proportion * (end_time - start_time)
        results.append({
            "text": s,
            "start": current_time,
            "end": current_time + duration
        })
        current_time += duration

    return results

# -----------------------
# Transcription endpoint (with optional split + first speech alignment)
# -----------------------
@app.post("/transcribe")
async def transcribe_whisper_only(
    video: UploadFile = File(...),
    with_timestamps: str = Form("0"),
    split_segments: str = Form("0")
):
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Save uploaded video
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        whisper_model = get_model()

        # 🔹 Detect actual first speech time
        first_speech_time = detect_first_speech(save_path)

        # Run transcription
        segments, info = whisper_model.transcribe(
            save_path,
            beam_size=5,
            initial_prompt=None,
            condition_on_previous_text=False,
        )

        transcription = []

        first_adjusted = False  # only adjust the first sentence once

        for seg in segments:
            if split_segments == "1":
                # 🔹 Split long segment into smaller sentences
                subs = split_segment_text_precise(seg.text, seg.start, seg.end)
                for i, sub in enumerate(subs):
                    entry = {"text": sub["text"].strip()}  # ✅ strip whitespace

                    s_start, s_end = sub["start"], sub["end"]

                    # Adjust first sentence start to detected speech offset
                    if not first_adjusted and first_speech_time > 0:
                        s_start = first_speech_time
                        first_adjusted = True

                    if with_timestamps == "1":
                        entry["start"] = seconds_to_hms(s_start)
                        entry["end"] = seconds_to_hms(s_end)

                    transcription.append(entry)
            else:
                # 🔹 Default: use Whisper segments directly
                entry = {"text": seg.text.strip()}  # ✅ strip whitespace

                s_start, s_end = seg.start, seg.end

                # Adjust only the very first segment
                if not first_adjusted and first_speech_time > 0:
                    s_start = first_speech_time
                    first_adjusted = True

                if with_timestamps == "1":
                    entry["start"] = seconds_to_hms(s_start)
                    entry["end"] = seconds_to_hms(s_end)

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
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Uvicorn on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
