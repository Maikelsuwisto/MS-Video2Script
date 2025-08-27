import time
import whisper
import os

_model_cache = {"model": None}

# Build absolute path so it's always correct inside container
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tiny.pt")

def load_model_cached():
    if _model_cache["model"] is None:
        _model_cache["model"] = whisper.load_model(MODEL_PATH)
    return _model_cache["model"]

def format_timestamp(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02}:{m:02}:{s:02}"

def generate_sentence_segments(segments, with_timestamp=True):
    """
    Split transcription into sentences.
    Returns a list of dicts: [{"start":..., "end":..., "text":...}, ...]
    """
    sentences = []
    for seg in segments:
        start_time = seg['start']
        end_time = seg['end']
        text = seg['text'].strip()
        if with_timestamp:
            sentences.append({
                "start": format_timestamp(start_time),
                "end": format_timestamp(end_time),
                "text": text
            })
        else:
            sentences.append({"text": text})
    return sentences

def transcribe_video(video_path, include_timestamps=True):
    """
    Transcribe video using Whisper.
    Returns a list of sentences.
    """
    model = load_model_cached()
    result = model.transcribe(video_path)
    return generate_sentence_segments(result["segments"], with_timestamp=include_timestamps)
