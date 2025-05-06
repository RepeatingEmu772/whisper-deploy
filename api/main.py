from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import shutil
import uvicorn

# Load model once at startup
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"  # or "int8_float16" if needed
)

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Run transcription
    segments, _ = model.transcribe(tmp_path, vad_filter=True)

    # Format results
    response = [{
        "start": seg.start,
        "end": seg.end,
        "text": seg.text
    } for seg in segments]

    return {"segments": response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
