import os, re
import uuid
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

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
# First speech detection
# -----------------------
def detect_first_speech(audio_path, min_speech_duration=1.0, threshold=0.6, lead_in_skip=0.25,
                        rms_window=0.05, rms_threshold=0.02):
    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np

    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    get_speech_timestamps, _, _, _, _ = utils

    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    wav_tensor = torch.from_numpy(wav).float()
    if sr != 16000:
        wav_tensor = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav_tensor)
        sr = 16000

    speech_timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=sr, threshold=threshold)

    for seg in speech_timestamps:
        dur = (seg['end'] - seg['start']) / sr
        if dur >= min_speech_duration:
            start_idx = seg['start'] + int(lead_in_skip * sr)
            end_idx = seg['end']
            window_size = int(rms_window * sr)

            for i in range(start_idx, end_idx - window_size, window_size):
                window = wav_tensor[i:i + window_size]
                rms = torch.sqrt(torch.mean(window ** 2)).item()
                if rms >= rms_threshold:
                    return i / sr
            return seg['start'] / sr
    return 0.0

# -----------------------
# Helper: smooth splitting
# -----------------------

def split_segment_text_smooth(seg, max_chars=40, min_chars=30, min_duration=1.0, first_block_extra=0.5):
    """
    Split segment into readable lines:
    - Lines ≤ max_chars
    - Lines ≥ min_chars (merge small fragments with previous)
    - Split at punctuation, commas, 'and', or force split
    - Assign proportional timestamps
    """
    # 1. Split into sentences by punctuation
    sentences = re.split(r'(?<=[.!?]) +', seg["text"])
    sentences = [s.strip() for s in sentences if s.strip()]

    # 2. Split long sentences at commas or 'and' or force split
    split_sentences = []
    for s in sentences:
        if len(s) <= max_chars:
            split_sentences.append(s)
        else:
            parts = re.split(r', | and ', s)
            current_line = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if len(current_line) + len(part) + 1 <= max_chars:
                    current_line += (" " if current_line else "") + part
                else:
                    if len(current_line) >= min_chars:
                        split_sentences.append(current_line)
                        current_line = part
                    else:
                        if split_sentences:
                            split_sentences[-1] += " " + current_line
                        else:
                            current_line = part
            if current_line:
                # Force split if still too long
                while len(current_line) > max_chars:
                    chunk = current_line[:max_chars].rsplit(" ", 1)[0]
                    if not chunk:
                        chunk = current_line[:max_chars]
                    split_sentences.append(chunk.strip())
                    current_line = current_line[len(chunk):].strip()
                if current_line:
                    split_sentences.append(current_line)

    # 3. Merge tiny lines (< min_chars) with previous line
    merged_sentences = []
    for line in split_sentences:
        if merged_sentences and len(line) < min_chars:
            merged_sentences[-1] += " " + line
        else:
            merged_sentences.append(line)

    # 4. Assign timestamps proportionally with min_duration
    total_duration = seg["end"] - seg["start"]
    total_chars = sum(len(s) for s in merged_sentences)
    block_entries = []
    current_start = seg["start"]

    for idx, block in enumerate(merged_sentences):
        block_duration = total_duration * (len(block) / total_chars) if total_chars > 0 else total_duration / len(merged_sentences)
        block_duration = max(block_duration, min_duration)
        if idx == 0:
            block_duration += first_block_extra
        block_end = current_start + block_duration
        block_end = max(block_end, current_start + min_duration)
        block_entries.append((current_start, block_end, block))
        current_start = block_end

    return block_entries

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

        # 🔹 Detect when real speech starts (in seconds)
        first_speech_time = detect_first_speech(save_path)
        print(f"First speech detected at {first_speech_time:.2f}s")

        # Run transcription (starting from detected speech)
        segments, info = whisper_model.transcribe(
            save_path,
            beam_size=5,
            initial_prompt=None,
            condition_on_previous_text=False,
        )

        transcription = []
        for idx, seg in enumerate(segments):
            # Skip segments that end before speech starts
            if seg.end <= first_speech_time:
                continue

            # Shift the very first segment so it doesn't start at 0
            if idx == 0 and seg.start < first_speech_time:
                seg_start = first_speech_time
            else:
                seg_start = seg.start

            seg_dict = {"start": seg_start, "end": seg.end, "text": seg.text}
            split_blocks = split_segment_text_smooth(seg_dict)

            for block_start, block_end, block_text in split_blocks:
                entry = {"text": block_text}
                if with_timestamps == "1":
                    entry["start"] = seconds_to_hms(block_start)
                    entry["end"] = seconds_to_hms(block_end)
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
